import os
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool
from livekit.plugins import openai, silero

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("food-delivery-agent")


class FoodDeliveryAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a support assistant for a food delivery app called QuickBite. "
                "Help customers track orders and resolve issues."
            )
        )

    @function_tool
    async def get_order_status(self, order_id: str) -> str:
        """Get the current status of a food delivery order."""
        logger.info(f"[Tool Invoked]: get_order_status(order_id='{order_id}')")

        # mocked order data
        orders = {
            "123": "Your order is being prepared and will arrive in 20 minutes.",
            "456": "Your order is out for delivery — driver is 5 minutes away!",
            "789": "Your order has been delivered. Enjoy your meal!",
        }
        result = orders.get(order_id, f"Order {order_id} not found. Please check your order ID.")
        logger.info(f"[Tool Result]: {result}")
        return result

    @function_tool
    async def cancel_order(self, order_id: str, reason: str) -> str:
        """Cancel a food delivery order."""
        logger.info(f"[Tool Invoked]: cancel_order(order_id='{order_id}', reason='{reason}')")
        result = f"Order {order_id} has been cancelled. Reason: {reason}. Refund will be processed in 3-5 days."
        logger.info(f"[Tool Result]: {result}")
        return result


async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting worker to LiveKit room: {ctx.room.name}")
    await ctx.connect()

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    session = AgentSession(
        llm=openai.LLM(
            model="openai/gpt-3.5-turbo",
            base_url=base_url,
            api_key=api_key,
        ),
        stt=openai.STT(),
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
