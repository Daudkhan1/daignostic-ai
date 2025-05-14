from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient


async def init_db():
    print("Initializing database connection")
    client = AsyncIOMotorClient(MongoDBConfig.MONGODB_URI)
    await init_beanie(
        database=client[MongoDBConfig.MONGODB_DB],
        document_models=[
            SupportRequest,
            User,
            Thread,
            Version,
            Document,
            ChatStore,
            DocumentStore,
        ],
    )
