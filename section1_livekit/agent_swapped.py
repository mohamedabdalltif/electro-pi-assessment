import os
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins import openai, deepgram, silero
from agent import FoodDeliveryAgent

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("food-delivery-agent-swapped")


async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting swapped worker to LiveKit room: {ctx.room.name}")
    await ctx.connect()

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    # Same agent, same LLM — only STT is swapped to Deepgram Nova-2
    session = AgentSession(
        llm=openai.LLM(
            model="openai/gpt-3.5-turbo",
            base_url=base_url,
            api_key=api_key,
        ),
        stt=deepgram.STT(model="nova-2"),
        tts=openai.TTS(voice="alloy"),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=FoodDeliveryAgent(),
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help with their order today."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
