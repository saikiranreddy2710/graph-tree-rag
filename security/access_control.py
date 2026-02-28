"""
RBAC Access Control — Role-Based Access Control for document-level security.
Filters retrieval results based on user roles and document access tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    RESTRICTED = "restricted"
    GUEST = "guest"


@dataclass
class User:
    user_id: str
    role: Role
    allowed_sources: Optional[set[str]] = None  # None = access all
    denied_sources: set[str] = field(default_factory=set)


@dataclass
class DocumentACL:
    """Access control list for a document."""

    doc_id: str
    allowed_roles: set[Role] = field(default_factory=lambda: {Role.ADMIN, Role.USER})
    allowed_users: Optional[set[str]] = None  # None = all users with right role
    classification: str = "internal"  # public, internal, confidential, restricted


class AccessController:
    """Enforces RBAC on retrieval results."""

    def __init__(self, enabled: bool = settings.security_rbac_enabled):
        self.enabled = enabled
        self._document_acls: dict[str, DocumentACL] = {}

    def register_document(self, acl: DocumentACL) -> None:
        """Register access control for a document."""
        self._document_acls[acl.doc_id] = acl

    def register_documents_bulk(
        self,
        doc_ids: list[str],
        allowed_roles: set[Role] = None,
        classification: str = "internal",
    ) -> None:
        """Register multiple documents with the same ACL."""
        if allowed_roles is None:
            allowed_roles = {Role.ADMIN, Role.USER}
        for doc_id in doc_ids:
            self._document_acls[doc_id] = DocumentACL(
                doc_id=doc_id,
                allowed_roles=allowed_roles,
                classification=classification,
            )

    def check_access(self, user: User, doc_id: str) -> bool:
        """Check if a user has access to a specific document."""
        if not self.enabled:
            return True

        # Admin always has access
        if user.role == Role.ADMIN:
            return True

        # Check denied sources
        if doc_id in user.denied_sources:
            return False

        # Check allowed sources whitelist
        if user.allowed_sources is not None and doc_id not in user.allowed_sources:
            return False

        # Check document ACL
        acl = self._document_acls.get(doc_id)
        if acl is None:
            # No ACL registered = default to role-based access
            return user.role in {Role.ADMIN, Role.USER}

        # Check role
        if user.role not in acl.allowed_roles:
            return False

        # Check user-level ACL
        if acl.allowed_users is not None and user.user_id not in acl.allowed_users:
            return False

        return True

    def filter_results(self, user: User, results: list) -> list:
        """Filter retrieval results based on user access."""
        if not self.enabled:
            return results

        filtered = []
        blocked_count = 0

        for result in results:
            doc_id = getattr(result, "doc_id", None) or ""
            metadata = getattr(result, "metadata", {})
            source = metadata.get("source", doc_id)

            # Check access using both doc_id and source
            if self.check_access(user, doc_id) and self.check_access(user, source):
                filtered.append(result)
            else:
                blocked_count += 1

        if blocked_count:
            logger.info(
                f"RBAC: filtered {blocked_count} results for user "
                f"{user.user_id} (role={user.role.value})"
            )

        return filtered
