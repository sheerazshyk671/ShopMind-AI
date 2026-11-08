"""Curated catalogue of a Pakistani online seller, priced from live market bands.

Every archetype below is anchored to a real 2026 Pakistani retail price band
(Daraz.pk / brand stores). The band midpoint becomes the base retail price; the
variants around it are what a real seller lists (Printed / Embroidered / Luxury,
Standard / Pro / Mini, and so on).

Sources for the price anchors and the seasonality/returns assumptions are listed
in ``data/demo/DATA_CARD.md``. Nothing here is scraped at runtime: the catalogue
is a fixed, auditable table so that a fixed seed reproduces a fixed dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Category economics
# ---------------------------------------------------------------------------
# return_rate      : COD-driven return / RTO rate. These were first set from published
#                    Pakistani bands (fashion 18-25%, beauty 12-18%, home 10-15%,
#                    electronics 8-12%) and are retained here as the *planning* assumption,
#                    because a seller should budget on a normal month rather than on a
#                    bad year. Measured on the real dataset (Pakistan's Largest E-Commerce
#                    Dataset), the realised rate is much higher - 48% overall, 65% for
#                    entertainment and 60% for mobiles - see docs/PAKISTAN_DATASET.md.
#                    The app exposes the assumption instead of hiding it.
# commission_rate  : marketplace commission. Published Daraz band: 8-15%.
# shipping_cost    : courier charge per delivered order (Rs), COD parcel.
# return_cost      : return-to-origin handling + packaging per returned unit (Rs).
# ad_rate          : marketing spend as a share of gross revenue.
# markup           : retail price / landed unit cost. Resale margin before fees.
# inflation        : monthly input-cost inflation applied to unit cost.
# items_per_order  : average units per order (affects shipping cost per unit).
CATEGORY_PROFILES = {
    'Fashion': {
        'return_rate': 0.21, 'commission_rate': 0.13, 'shipping_cost': 210,
        'return_cost': 180, 'ad_rate': 0.045, 'markup': 1.95,
        'cost_inflation': 0.006, 'price_inflation': 0.004, 'items_per_order': 1.25,
        # Pakistani retail calendar: Ramadan & both Eids drive apparel hardest.
        'seasonality': {1: 0.85, 2: 1.25, 3: 1.60, 4: 1.35, 5: 1.25, 6: 0.85,
                        7: 0.80, 8: 0.95, 9: 0.90, 10: 1.00, 11: 1.10, 12: 1.05},
        'base_units': 30.0,
    },
    'Electronics': {
        'return_rate': 0.10, 'commission_rate': 0.08, 'shipping_cost': 260,
        'return_cost': 320, 'ad_rate': 0.055, 'markup': 1.55,
        'cost_inflation': 0.012, 'price_inflation': 0.010, 'items_per_order': 1.10,
        # 11.11, Black Friday and Independence-Day sales carry Q4.
        'seasonality': {1: 0.90, 2: 1.00, 3: 1.05, 4: 1.00, 5: 1.00, 6: 1.05,
                        7: 1.10, 8: 1.15, 9: 1.00, 10: 1.05, 11: 1.45, 12: 1.20},
        'base_units': 16.0,
    },
    'Home & Kitchen': {
        'return_rate': 0.12, 'commission_rate': 0.11, 'shipping_cost': 300,
        'return_cost': 260, 'ad_rate': 0.040, 'markup': 1.75,
        'cost_inflation': 0.007, 'price_inflation': 0.005, 'items_per_order': 1.15,
        # Ramadan cooking, summer juicers, winter bedding.
        'seasonality': {1: 1.00, 2: 1.15, 3: 1.20, 4: 1.05, 5: 1.15, 6: 1.10,
                        7: 1.05, 8: 0.95, 9: 0.95, 10: 1.00, 11: 1.10, 12: 1.15},
        'base_units': 13.0,
    },
    'Beauty & Personal Care': {
        'return_rate': 0.15, 'commission_rate': 0.12, 'shipping_cost': 200,
        'return_cost': 170, 'ad_rate': 0.060, 'markup': 2.30,
        'cost_inflation': 0.009, 'price_inflation': 0.007, 'items_per_order': 1.45,
        # Wedding season + Eid + Black Friday.
        'seasonality': {1: 0.95, 2: 1.15, 3: 1.30, 4: 1.20, 5: 1.05, 6: 0.95,
                        7: 1.00, 8: 0.95, 9: 1.00, 10: 1.00, 11: 1.25, 12: 1.10},
        'base_units': 52.0,
    },
    'Baby & Kids': {
        'return_rate': 0.11, 'commission_rate': 0.10, 'shipping_cost': 210,
        'return_cost': 190, 'ad_rate': 0.035, 'markup': 2.00,
        'cost_inflation': 0.007, 'price_inflation': 0.005, 'items_per_order': 1.60,
        # Back-to-school in August.
        'seasonality': {1: 1.00, 2: 1.05, 3: 1.10, 4: 1.00, 5: 1.00, 6: 1.00,
                        7: 1.05, 8: 1.25, 9: 1.15, 10: 1.00, 11: 1.05, 12: 1.05},
        'base_units': 38.0,
    },
    'Sports & Fitness': {
        'return_rate': 0.13, 'commission_rate': 0.09, 'shipping_cost': 280,
        'return_cost': 240, 'ad_rate': 0.045, 'markup': 1.85,
        'cost_inflation': 0.008, 'price_inflation': 0.006, 'items_per_order': 1.20,
        # New-year resolutions, then a summer dip.
        'seasonality': {1: 1.35, 2: 1.15, 3: 1.00, 4: 1.00, 5: 0.95, 6: 0.90,
                        7: 0.90, 8: 0.95, 9: 1.00, 10: 1.05, 11: 1.05, 12: 1.10},
        'base_units': 20.0,
    },
}

VARIANT_LABELS = {
    'Fashion': ['Printed', 'Embroidered', 'Luxury'],
    'Electronics': ['Standard', 'Pro', 'Mini'],
    'Home & Kitchen': ['5-Piece', '7-Piece', 'Deluxe'],
    'Beauty & Personal Care': ['100ml', '200ml', 'Combo Pack'],
    'Baby & Kids': ['Small', 'Medium', 'Large'],
    'Sports & Fitness': ['Basic', 'Pro', 'Premium'],
}

VARIANT_PRICE_MULT = [0.85, 1.00, 1.35]
VARIANT_DEMAND_MULT = [1.15, 1.00, 0.70]


@dataclass(frozen=True)
class Archetype:
    """One product family a seller lists in three variants."""

    name: str
    category: str
    price_low: int
    price_high: int
    demand_scale: float = 1.0
    notes: str = field(default='', compare=False)
    variants: tuple[str, str, str] = ('Standard', 'Plus', 'Premium')


# Price bands verified against live Pakistani listings in November 2026.
ARCHETYPES: tuple[Archetype, ...] = (
    # --- Fashion -----------------------------------------------------------
    Archetype('Lawn Suit', 'Fashion', 2990, 7990, 1.25,
              'Gul Ahmed / Khaadi printed and embroidered lawn bands',
              ('Printed', 'Embroidered', 'Luxury')),
    Archetype('3-Piece Chiffon Suit', 'Fashion', 6990, 15990, 0.75,
              'Festive and wedding-season lawn/chiffon',
              ('Semi-Formal', 'Formal', 'Bridal')),
    Archetype('Stitched Cotton Kurti', 'Fashion', 1790, 3490, 1.35,
              'Ready-to-wear basics', ('Casual', 'Embroidered', 'Designer')),
    Archetype('Cotton Bedsheet Set', 'Fashion', 2290, 4490, 1.00,
              'Hometex / affordable.pk king-size band',
              ('Single', 'Double', 'King')),
    Archetype('Ladies PU Leather Handbag', 'Fashion', 1490, 3990, 1.10,
              'Daraz handbag band Rs 500-4,000', ('Shoulder', 'Tote', 'Clutch')),
    Archetype("Men's Cotton Kurta", 'Fashion', 2490, 4490, 1.10, '',
              ('Plain', 'Embroidered', 'Premium')),
    Archetype('Lawn Co-Ord Set', 'Fashion', 2790, 5490, 0.95, '',
              ('Printed', 'Embroidered', 'Luxury')),
    # --- Electronics -------------------------------------------------------
    Archetype('Power Bank', 'Electronics', 2490, 4590, 1.30,
              'Daraz power bank band Rs 900-4,600',
              ('10,000mAh', '20,000mAh', 'Fast Charge')),
    Archetype('TWS Wireless Earbuds', 'Electronics', 2990, 6990, 1.25,
              'Daraz earbuds band Rs 2,500-35,000 (low end)',
              ('Standard', 'ENC', 'Gaming')),
    Archetype('25W USB-C Fast Charger', 'Electronics', 1290, 2490, 1.45, '',
              ('20W', '25W', '65W')),
    Archetype('Mini Bluetooth Speaker', 'Electronics', 2290, 4490, 1.05, '',
              ('5W', '10W', 'Party')),
    Archetype('Basic Smart Watch', 'Electronics', 3990, 7990, 0.85, '',
              ('Basic', 'Fitness', 'AMOLED')),
    Archetype('LED Ring Light', 'Electronics', 1990, 3990, 0.95, '',
              ('10-Inch', '12-Inch', '18-Inch')),
    Archetype('USB-C Charging Cable', 'Electronics', 690, 1590, 1.60, '',
              ('1m', '2m', 'Braided')),
    # --- Home & Kitchen ----------------------------------------------------
    Archetype('Non-Stick Cookware Set', 'Home & Kitchen', 5490, 11990, 0.80,
              'Daraz 16-piece non-stick set band', ('7-Piece', '12-Piece', '16-Piece')),
    Archetype('Air Fryer', 'Home & Kitchen', 17900, 34900, 0.35,
              'Anex / Westpoint / Kenwood band Rs 17,900-55,900',
              ('5L', '8L', 'Dual Basket')),
    Archetype('Electric Kettle', 'Home & Kitchen', 5900, 9900, 0.75,
              'Anex kettle band Rs 8,400-9,300', ('1.5L', '1.8L', '2.0L')),
    Archetype('2-in-1 Juicer Blender', 'Home & Kitchen', 6900, 14900, 0.60,
              'Blender band Rs 4,000-35,000 (mid)',
              ('2-in-1', '3-in-1', '4-in-1')),
    Archetype('Stainless Steel Water Bottle', 'Home & Kitchen', 890, 1790, 1.50, '',
              ('500ml', '750ml', '1L')),
    Archetype('Kitchen Storage Organizer', 'Home & Kitchen', 1290, 2790, 1.20, '',
              ('2-Piece', '3-Piece', '5-Piece')),
    Archetype('Dinner Set', 'Home & Kitchen', 4990, 8990, 0.55, '',
              ('18-Piece', '24-Piece', '32-Piece')),
    Archetype('Bedside Table Lamp', 'Home & Kitchen', 1790, 3490, 0.85, '',
              ('Basic', 'Dimmable', 'Touch')),
    # --- Beauty & Personal Care -------------------------------------------
    Archetype('Herbal Face Wash', 'Beauty & Personal Care', 490, 1190, 1.60, '',
              ('50ml', '100ml', 'Combo')),
    Archetype('Vitamin C Serum', 'Beauty & Personal Care', 990, 2390, 1.30, '',
              ('15ml', '30ml', 'Duo Pack')),
    Archetype('Sunblock SPF 50', 'Beauty & Personal Care', 790, 1790, 1.35, '',
              ('50ml', '75ml', 'Family Pack')),
    Archetype('Herbal Hair Oil', 'Beauty & Personal Care', 490, 1090, 1.30, '',
              ('100ml', '200ml', 'Combo')),
    Archetype('Perfume', 'Beauty & Personal Care', 1490, 3490, 1.00, '',
              ('30ml', '50ml', 'Gift Set')),
    Archetype('Makeup Brush Set', 'Beauty & Personal Care', 990, 2490, 1.05, '',
              ('6-Piece', '12-Piece', '24-Piece')),
    Archetype('Lipstick Matte', 'Beauty & Personal Care', 390, 990, 1.55, '',
              ('Single', 'Duo', 'Gift Set')),
    Archetype('Aloe Vera Gel', 'Beauty & Personal Care', 490, 1190, 1.25, '',
              ('100ml', '200ml', '500ml')),
    # --- Baby & Kids -------------------------------------------------------
    Archetype('Baby Diaper Pants (Pack)', 'Baby & Kids', 1290, 2490, 1.45, '',
              ('Small', 'Medium', 'Large')),
    Archetype('Kids Water Bottle', 'Baby & Kids', 690, 1490, 1.30, '',
              ('400ml', '600ml', '800ml')),
    Archetype('Baby Fleece Blanket', 'Baby & Kids', 990, 2190, 0.95, '',
              ('Small', 'Medium', 'Large')),
    Archetype('Kids Lunch Box Set', 'Baby & Kids', 890, 1890, 1.15, '',
              ('2-Piece', '3-Piece', '5-Piece')),
    Archetype('Baby Bath Gift Set', 'Baby & Kids', 1490, 3290, 0.85, '',
              ('Basic', 'Deluxe', 'Premium')),
    # --- Sports & Fitness --------------------------------------------------
    Archetype('Yoga Mat', 'Sports & Fitness', 1290, 2790, 1.00, '',
              ('4mm', '6mm', '8mm')),
    Archetype('Dumbbell Set', 'Sports & Fitness', 3990, 7490, 0.50, '',
              ('5kg', '10kg', '20kg')),
    Archetype('Skipping Rope', 'Sports & Fitness', 390, 890, 1.35, '',
              ('Basic', 'Counter', 'Speed')),
    Archetype('Resistance Band Set', 'Sports & Fitness', 890, 1990, 1.05, '',
              ('3-Piece', '5-Piece', '7-Piece')),
    Archetype('Adjustable Hand Gripper', 'Sports & Fitness', 590, 1290, 1.10, '',
              ('Basic', 'Adjustable', 'Pro')),
)


def category_profile(category: str) -> dict:
    """Return the economics profile for a category, failing loudly on typos."""
    try:
        return CATEGORY_PROFILES[category]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f'Unknown category {category!r}') from exc


def seasonal_index(category: str, calendar_month: int) -> float:
    """Multiplicative demand index for a category in a calendar month (1-12)."""
    return float(category_profile(category)['seasonality'][int(calendar_month)])


def price_band_midpoint(archetype: Archetype) -> float:
    """Geometric midpoint of the verified price band (robust to long tails)."""
    return float((archetype.price_low * archetype.price_high) ** 0.5)
