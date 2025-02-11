from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel


class ReservationDecision(str, Enum):
    NO_COUPON_RESERVATION = "no_coupon_reservation"
    COUPON_RESERVATION = "coupon_reservation"
    NO_RESERVATION = "no_reservation"


class ReservationResponse(BaseModel):
    decision: ReservationDecision


def perform_user_behavior(
    coupon_df: pd.DataFrame,
    user_df: pd.DataFrame,
    restaurant_df: pd.DataFrame,
    client: Any,
    client_model: str,
    temperature: float,
    test_mode: bool = False,  # test mode if True, else for validation  # TODO: impl. test_mode
    sample_ratio: float = 0.03,
) -> dict[str, Any]:
    """Simulate the behavior of personas reacting to coupon allocation using an LLM.

    Returns aggregated metrics such as lift and ROI.
    """
    total_revenue_with_coupon = 0
    total_cost = 0
    total_revenue_without_coupon = 0

    # NOTE: We can apply clustering method to group users based on their behavior for better simulation
    sampled_users = user_df.sample(frac=sample_ratio, random_state=42)
    user_dict = sampled_users.set_index("user_id").to_dict(orient="index")

    # LLM-based decision making
    for _, coupon in coupon_df.iterrows():
        user_id = coupon["user_id"]
        if user_id not in user_dict:
            continue
        user = user_dict[user_id]
        restaurant = restaurant_df[
            restaurant_df["restaurant_id"] == coupon["restaurant_id"]
        ].iloc[0]

        # FIXME: Use pydantic structured output to not rely on string parsing.
        # For simplicity, assuming the user is only interested in the restaurant where the coupon is allocated.
        system_prompt = """
        You are a simulated customer looking for a restaurant to dine in.
        You have received a coupon for the restaurant you have checked on a booking site.
        Based on your profile and preferences, decide whether you would book a reservation with the given coupon.
        """
        user_behavior_prompt = f"""
        **Your Profile:**
        - Solo visits: {user["usage_1_person"]} yen/year
        - With a partner: {user["usage_2_people"]} yen/year
        - With a group (3-4 people): {user["usage_3-4_people"]} yen/year
        - With 5+ people: {user["usage_5_or_more_people"]} yen/year
        - Average lunch spending: ${user["avg_lunch_price"]} yen/time
        - Average dinner spending: ${user["avg_dinner_price"]} yen/time

        **Decision Rules:**
        1. `"no_coupon_reservation"` → You would have booked **even without a coupon**.
        2. `"coupon_reservation"` → You **will book only because of the coupon**.
        3. `"no_reservation"` → You will **not book even with a coupon**.
        
        **Coupon & Restaurant Details:**
        - coupon amount: {coupon["coupon_amount"]} yen
        - minimum spending: {coupon["min_spending"]} yen
        - restaurant
            - average lunch price: {restaurant["lunch_price"]} yen
            - average dinner price: {restaurant["dinner_price"]} yen
            - last month sales 1 person: {restaurant["last_month_sales_1_person"]} yen
            - last month sales 2 people: {restaurant["last_month_sales_2_people"]} yen
            - last month sales 3-4 people: {restaurant["last_month_sales_3-4_people"]} yen
            - last month sales 5+ people: {restaurant["last_month_sales_5_or_more"]} yen
        """

        # NOTE: openai only
        response = client.beta.chat.completions.parse(
            model=client_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_behavior_prompt},
            ],
            temperature=temperature,
            response_format=ReservationResponse,
        )
        response = response.choices[0].message.parsed
        print(f"DEBUG: User {user_id} decision: {response.decision}")

        # For simplicity, define the lift as follows (ignoring logical correctness here)
        # lift = (revenue_with_coupon - revenue_without_coupon
        # ROI = (lift - cost) / cost
        if response.decision == ReservationDecision.COUPON_RESERVATION:
            # Assume 20% above the minimum spending
            total_revenue_with_coupon += coupon["min_spending"] * 1.2
            total_cost += coupon["coupon_amount"]
        elif response.decision == ReservationDecision.NO_COUPON_RESERVATION:
            total_revenue_without_coupon += coupon["min_spending"] * 1.2

    # FIXME: use cluster information to simulate other user behaviors
    total_revenue_with_coupon *= 1 / sample_ratio
    total_revenue_without_coupon *= 1 / sample_ratio
    total_cost *= 1 / sample_ratio

    lift = total_revenue_with_coupon - total_revenue_without_coupon
    roi = (lift - total_cost) / total_cost
    return {"roi": roi, "lift": lift}
