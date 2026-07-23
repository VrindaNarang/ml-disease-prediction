"""Mock data seeder for local/dev use (Milestone 4: Sales Analytics Dashboard).

Generates a realistic-but-modest dataset (colleges, products, sales reps,
orders, order items) so the dashboard/analytics/customers/products APIs have
real aggregate data to serve instead of hardcoded responses. This is
deliberately smaller than the 500-college / 20k-order target described in
PROJECT_SPEC.md section 13 (which belongs to the full ML milestone's data
generator under ml/data_generation/) — it exists purely to back this
milestone's CRUD + analytics endpoints.

Run with:  python -m app.db.seed [--reset]
"""

from __future__ import annotations

import argparse
import datetime as dt
import random

from faker import Faker
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models import College, Order, OrderItem, Product, ProductCategory, SalesRep

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)

STATES_BY_REGION: dict[str, list[str]] = {
    "North": ["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Rajasthan"],
    "South": ["Karnataka", "Tamil Nadu", "Andhra Pradesh", "Telangana", "Kerala"],
    "East": ["West Bengal", "Odisha", "Bihar", "Jharkhand"],
    "West": ["Maharashtra", "Gujarat", "Madhya Pradesh", "Goa"],
}

CITIES_BY_STATE: dict[str, list[str]] = {
    "Delhi": ["New Delhi", "Dwarka", "Rohini"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Noida"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Tirupati"],
    "Telangana": ["Hyderabad", "Warangal"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode"],
    "West Bengal": ["Kolkata", "Siliguri", "Durgapur"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela"],
    "Bihar": ["Patna", "Gaya"],
    "Jharkhand": ["Ranchi", "Jamshedpur"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior"],
    "Goa": ["Panaji", "Margao"],
}

COLLEGE_NAME_TEMPLATES = [
    "{city} Government Polytechnic",
    "{city} Institute of Technology",
    "Government College of Engineering, {city}",
    "{city} Medical College",
    "St. Xavier's College, {city}",
    "{city} University Department of Chemistry",
    "National Institute of Technology, {city}",
    "{city} College of Pharmacy",
    "Rashtriya Vidyalaya, {city}",
    "{city} Institute of Applied Sciences",
    "Shri Ram College of Science, {city}",
    "{city} Women's College",
    "Indian Institute of Science Education, {city}",
    "{city} Polytechnic for Women",
    "Sacred Heart College, {city}",
]

CATEGORY_SPECS: list[tuple[str, str, tuple[int, int], list[str]]] = [
    (
        "Glassware",
        "Beakers, flasks, test tubes, and general laboratory glassware",
        (150, 3500),
        [
            "Borosilicate Beaker Set",
            "Erlenmeyer Flask",
            "Graduated Cylinder",
            "Test Tube Rack",
            "Volumetric Flask",
            "Petri Dish Pack",
            "Watch Glass Set",
            "Burette (Class A)",
            "Glass Funnel Set",
            "Reagent Bottle (Amber)",
            "Round Bottom Flask",
            "Desiccator",
        ],
    ),
    (
        "Reagents & Chemicals",
        "Analytical and reagent-grade chemicals",
        (200, 8000),
        [
            "Sulfuric Acid (AR Grade)",
            "Sodium Hydroxide Pellets",
            "Ethanol (Absolute)",
            "Hydrochloric Acid",
            "Potassium Permanganate",
            "Acetone (AR Grade)",
            "Copper Sulfate Crystals",
            "Silver Nitrate",
            "Phenolphthalein Indicator",
            "Distilled Water (5L Can)",
            "Sodium Chloride (AR Grade)",
            "Methanol (HPLC Grade)",
        ],
    ),
    (
        "Analytical Instruments",
        "pH meters, spectrophotometers, balances, and benchtop instruments",
        (2500, 150000),
        [
            "Digital pH Meter",
            "UV-Vis Spectrophotometer",
            "Analytical Balance (0.1mg)",
            "Benchtop Centrifuge",
            "Hot Air Oven",
            "Muffle Furnace",
            "Magnetic Stirrer with Hotplate",
            "Colony Counter",
            "Digital Water Bath",
            "Autoclave (Vertical)",
            "Refractometer",
            "Conductivity Meter",
        ],
    ),
    (
        "Safety Equipment",
        "Personal protective equipment and lab safety fixtures",
        (100, 25000),
        [
            "Nitrile Gloves (Box of 100)",
            "Safety Goggles",
            "Lab Coat (Cotton)",
            "Fume Hood (Ductless)",
            "CO2 Fire Extinguisher",
            "Eye Wash Station",
            "First Aid Kit",
            "Face Shield",
            "Chemical Splash Apron",
            "Spill Containment Kit",
        ],
    ),
    (
        "Consumables",
        "Filter paper, pipette tips, labels, and other lab consumables",
        (50, 2000),
        [
            "Filter Paper (Whatman Grade 1)",
            "Micropipette Tips (Box)",
            "Parafilm Roll",
            "Disposable Petri Dishes (Pack)",
            "Cotton Wool Roll",
            "Litmus Paper Strips",
            "Weighing Boats (Pack)",
            "Sample Vials (Pack of 50)",
            "Lab Labels Roll",
            "Disposable Syringes (Pack)",
        ],
    ),
    (
        "Lab Furniture",
        "Workbenches, storage cabinets, and lab fit-out furniture",
        (3000, 60000),
        [
            "Laboratory Workbench",
            "Chemical Storage Cabinet",
            "Adjustable Lab Stool",
            "Fume Cupboard",
            "Glassware Storage Rack",
            "Mobile Lab Trolley",
            "Wall-mounted Shelf Unit",
            "Acid-resistant Cabinet",
        ],
    ),
]

UNITS = ["piece", "box", "pack", "litre", "kg", "set", "roll"]
SIZE_VARIANTS = ["", " (Small)", " (250ml)", " (500ml)", " (1L)", " (Large)", " (5L)", " (Pack of 10)"]

SALES_REP_COUNT = 8
COLLEGE_COUNT = 60
PRODUCT_TARGET = 85
ORDER_MONTHS = 24


def _weighted_choice(options: list[tuple[float, object]]) -> object:
    total = sum(weight for weight, _ in options)
    pick = random.uniform(0, total)
    upto = 0.0
    for weight, value in options:
        upto += weight
        if pick <= upto:
            return value
    return options[-1][1]


def seed_categories(db: Session) -> list[ProductCategory]:
    categories = []
    for name, description, _price_band, _names in CATEGORY_SPECS:
        category = ProductCategory(name=name, description=description)
        db.add(category)
        categories.append(category)
    db.flush()
    return categories


def seed_products(db: Session, categories: list[ProductCategory]) -> list[Product]:
    products: list[Product] = []
    sku_counter = 1000
    for category, (_, _, price_band, names) in zip(categories, CATEGORY_SPECS):
        prefix = "".join(w[0] for w in category.name.split())[:3].upper()
        variants_needed = max(1, PRODUCT_TARGET // len(CATEGORY_SPECS))
        combos = [(name, size) for name in names for size in SIZE_VARIANTS]
        random.shuffle(combos)
        for name, size in combos[:variants_needed]:
            sku_counter += 1
            unit_price = round(random.uniform(*price_band), 2)
            cost_price = round(unit_price * random.uniform(0.55, 0.78), 2)
            product = Product(
                sku=f"{prefix}-{sku_counter}",
                name=f"{name}{size}",
                category_id=category.id,
                unit_price=unit_price,
                cost_price=cost_price,
                unit_of_measure=random.choice(UNITS),
                is_active=random.random() > 0.05,
            )
            db.add(product)
            products.append(product)
    db.flush()
    return products


def seed_sales_reps(db: Session) -> list[SalesRep]:
    reps = []
    for _ in range(SALES_REP_COUNT):
        region = random.choice(list(STATES_BY_REGION))
        name = fake.name()
        rep = SalesRep(
            name=name,
            email=f"{name.lower().replace(' ', '.')}@salespilot-reps.example.com",
            region=region,
            hire_date=fake.date_between(start_date="-6y", end_date="-1y"),
        )
        db.add(rep)
        reps.append(rep)
    db.flush()
    return reps


def seed_colleges(db: Session) -> list[College]:
    colleges = []
    used_names: set[str] = set()
    today = dt.date.today()
    for _ in range(COLLEGE_COUNT):
        region = random.choice(list(STATES_BY_REGION))
        state = random.choice(STATES_BY_REGION[region])
        city = random.choice(CITIES_BY_STATE[state])

        name = None
        for _attempt in range(10):
            candidate = random.choice(COLLEGE_NAME_TEMPLATES).format(city=city)
            if candidate not in used_names:
                name = candidate
                used_names.add(candidate)
                break
        if name is None:
            name = f"{city} Institute #{len(colleges) + 1}"

        institution_type = _weighted_choice([(0.6, "government"), (0.4, "private")])
        is_dormant = random.random() < 0.2
        status = "dormant" if is_dormant else "active"

        college = College(
            name=name,
            institution_type=institution_type,
            region=region,
            state=state,
            city=city,
            address=fake.street_address(),
            contact_name=fake.name(),
            contact_email=fake.company_email(),
            contact_phone=fake.phone_number()[:30],
            onboarded_date=fake.date_between(start_date="-6y", end_date="-2y"),
            status=status,
        )
        db.add(college)
        colleges.append(college)
    db.flush()
    return colleges


def _order_cadence_profile() -> str:
    return _weighted_choice(
        [
            (0.25, "frequent"),
            (0.40, "regular"),
            (0.35, "sporadic"),
        ]
    )


def _seasonal_weight(month: int) -> float:
    # Academic-year start (Jun-Jul) and grant/fiscal cycle (Feb-Mar) peaks.
    if month in (6, 7):
        return 1.8
    if month in (2, 3):
        return 1.5
    if month in (11,):
        return 1.3
    if month in (12, 1):
        return 0.7
    return 1.0


def seed_orders(
    db: Session, colleges: list[College], products: list[Product], reps: list[SalesRep]
) -> None:
    today = dt.date.today()
    # Normalized to day=1 so month-to-month comparisons below aren't skewed
    # by the day-of-month the script happens to run on.
    window_start = (today - dt.timedelta(days=ORDER_MONTHS * 30)).replace(day=1)
    order_counter = 1
    reps_by_region: dict[str, list[SalesRep]] = {}
    for rep in reps:
        reps_by_region.setdefault(rep.region, []).append(rep)

    active_products = [p for p in products if p.is_active]

    for college in colleges:
        profile = _order_cadence_profile()
        base_orders_per_month = {"frequent": 2.2, "regular": 0.9, "sporadic": 0.3}[profile]

        if college.status == "dormant":
            cutoff_days_ago = random.randint(190, 480)
            college_last_order = today - dt.timedelta(days=cutoff_days_ago)
        else:
            college_last_order = today

        region_reps = reps_by_region.get(college.region) or reps
        rep = random.choice(region_reps)

        month_cursor = window_start
        while month_cursor <= today:
            if month_cursor > college_last_order:
                break
            weight = _seasonal_weight(month_cursor.month)
            expected_orders = base_orders_per_month * weight
            n_orders = max(0, round(random.gauss(expected_orders, expected_orders * 0.4)))

            for _ in range(n_orders):
                day_offset = random.randint(0, 27)
                order_date = month_cursor.replace(day=1) + dt.timedelta(days=day_offset)
                if order_date > today or order_date > college_last_order:
                    continue

                n_items = random.randint(1, 6)
                chosen_products = random.sample(
                    active_products, k=min(n_items, len(active_products))
                )

                order_discount = round(random.uniform(0, 5), 2)
                if college.institution_type == "government" and random.random() < 0.4:
                    order_discount = round(order_discount + random.uniform(2, 6), 2)

                items_payload = []
                subtotal = 0.0
                for product in chosen_products:
                    price = float(product.unit_price)
                    if price > 20_000:
                        quantity = random.randint(1, 3)
                    elif price > 3_000:
                        quantity = random.randint(1, 8)
                    else:
                        quantity = random.randint(2, 40)
                    line_discount = round(random.uniform(0, 3), 2)
                    line_total = round(
                        float(product.unit_price) * quantity * (1 - line_discount / 100), 2
                    )
                    subtotal += line_total
                    items_payload.append((product, quantity, line_discount, line_total))

                subtotal = round(subtotal, 2)
                tax_amount = round(subtotal * 0.18, 2)
                total_amount = round(subtotal + tax_amount, 2)

                if college.institution_type == "government":
                    payment_status = _weighted_choice(
                        [(0.45, "paid"), (0.30, "pending"), (0.25, "overdue")]
                    )
                else:
                    payment_status = _weighted_choice(
                        [(0.70, "paid"), (0.22, "pending"), (0.08, "overdue")]
                    )

                payment_due_date = order_date + dt.timedelta(days=30)
                payment_received_date = None
                if payment_status == "paid":
                    delay = random.randint(-5, 45) if college.institution_type == "government" else random.randint(-5, 20)
                    payment_received_date = min(payment_due_date + dt.timedelta(days=delay), today)
                    if payment_received_date < order_date:
                        payment_received_date = order_date

                order = Order(
                    order_number=f"ORD-{order_date.year}-{order_counter:06d}",
                    college_id=college.id,
                    sales_rep_id=rep.id,
                    order_date=order_date,
                    status="fulfilled" if payment_status != "overdue" else random.choice(["fulfilled", "pending"]),
                    payment_status=payment_status,
                    payment_due_date=payment_due_date,
                    payment_received_date=payment_received_date,
                    discount_pct=order_discount,
                    subtotal=subtotal,
                    tax_amount=tax_amount,
                    total_amount=total_amount,
                )
                db.add(order)
                db.flush()

                for product, quantity, line_discount, line_total in items_payload:
                    db.add(
                        OrderItem(
                            order_id=order.id,
                            product_id=product.id,
                            quantity=quantity,
                            unit_price=product.unit_price,
                            discount_pct=line_discount,
                            line_total=line_total,
                        )
                    )
                order_counter += 1

            # advance to next month
            if month_cursor.month == 12:
                month_cursor = month_cursor.replace(year=month_cursor.year + 1, month=1)
            else:
                month_cursor = month_cursor.replace(month=month_cursor.month + 1)

    db.flush()
    print(f"Seeded {order_counter - 1} orders")


def reset_data(db: Session) -> None:
    db.execute(text("TRUNCATE TABLE order_items, orders, products, product_categories, sales_reps, colleges RESTART IDENTITY CASCADE"))
    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Truncate existing data before seeding")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        existing = db.query(College).count()
        if existing and not args.reset:
            print(f"Database already has {existing} colleges. Pass --reset to re-seed.")
            return
        if args.reset:
            reset_data(db)

        print("Seeding product categories & products...")
        categories = seed_categories(db)
        products = seed_products(db, categories)
        print(f"  {len(categories)} categories, {len(products)} products")

        print("Seeding sales reps...")
        reps = seed_sales_reps(db)
        print(f"  {len(reps)} sales reps")

        print("Seeding colleges...")
        colleges = seed_colleges(db)
        print(f"  {len(colleges)} colleges")

        print("Seeding orders (this can take a little while)...")
        seed_orders(db, colleges, products, reps)

        db.commit()
        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
