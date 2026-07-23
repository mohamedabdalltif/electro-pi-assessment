import os
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins.google.realtime import RealtimeModel
from agent import FoodDeliveryAgent

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("food-delivery-agent-gemini")


async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting Gemini worker to LiveKit room: {ctx.room.name}")
    await ctx.connect()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your env.")
        return

    # Using the multimodal realtime model directly for audio input/output
    llm = RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key=api_key,
        voice="Aoede",
        temperature=0.7,
        instructions=(
            "You are a support assistant for a food delivery app called QuickBite. "
            "Help customers track orders and resolve issues."
        )
    )

    session = AgentSession(llm=llm)

    await session.start(
        room=ctx.room,
        agent=FoodDeliveryAgent(),
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help with their order today."
    )


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
