"""
Manually register a user against the configured database.
"""

import asyncio
from getpass import getpass

from document_rag_service.core.settings import get_settings
from document_rag_service.db.session import (
    create_database_engine,
    create_session_factory,
)
from document_rag_service.services.user_service import UserService


async def main() -> None:
    """
    Prompt for credentials, register one user, and close database resources.
    """

    settings = get_settings()
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        email = input("Email: ").strip()
        password = getpass("Password: ")

        async with session_factory() as session:
            user_service = UserService(session)

            user = await user_service.register_user(
                email=email,
                password=password,
            )

            print("\nUSER CREATED")
            print("id=", user.id)
            print("email=", user.email)
            print("active=", user.is_active)

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
