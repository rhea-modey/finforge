"""Seeded name pools for FinForge world generation. Pure data + picker helpers."""

COMPANY_NAMES = {
    "saas": ["Cirrus Metrics Inc.", "Beacon Ledger Software", "Northlight Systems Inc.",
             "Fathom Analytics Co.", "Quartz Relay Inc.", "Halcyon Grid Software",
             "Vantage Loop Inc.", "Bluewater Stack Co.", "Ember Signal Inc."],
    "ecommerce": ["Juniper Goods Co.", "Copper Kettle Trading", "Wildwood Supply Co.",
                  "Meridian Basket Inc.", "Larkspur Home Goods", "Tidewater Outfitters",
                  "Maple & Main Retail Co.", "Foxglove Market Inc.",
                  "Saltbox Mercantile Co."],
    "wholesale": ["Granite Peak Distribution", "Bluebird Wholesale Co.", "Cascade Supply Partners",
                  "Ironline Distributors Inc.", "Prairie State Goods Co.", "Keystone Freight & Supply",
                  "Harborline Supply Co."],
    "services": ["Harborview Consulting LLP", "Redstone Advisory Group", "Clearpath Partners LLC",
                 "Summit Ridge Advisors", "Lakefront Strategy Group", "Stonebridge Professional Services",
                 "Ashworth & Bell Advisors", "Foxhall Consulting Group"],
    "manufacturing": ["Anvil Works Manufacturing", "Foundry Line Industries", "Precision Arc Fabrication",
                      "Millbrook Components Inc.", "Sturgeon Bay Machining Co."],
    "clinic": ["Lakeside Family Health LLC", "Cedar Grove Medical Group", "Riverbend Wellness Clinic",
               "Oak Street Pediatrics LLC", "Harbor Point Physical Therapy"],
}

CUSTOMER_POOL = [
    "Atlas Verde LLC", "Brockman & Hale", "Crestline Partners", "Dunmore Holdings",
    "Eastgate Collective", "Ferrell Industries", "Glenrock Solutions", "Hartwell Group",
    "Ivywood Enterprises", "Jasper Analytics", "Kelso Brands", "Longmont Systems",
    "Marlowe & Finch", "Nordica Labs", "Ostrander Co.", "Pinehurst Digital",
    "Quimby Ventures", "Rosewood Retail", "Silverthorne Media", "Tamarack Health",
    "Underhill Logistics", "Vesper Technologies", "Wexford Group", "Yardley Supply",
    "Zephyr Dynamics", "Aldergate Insurance", "Birchfield Foods", "Corliss Marine",
]

VENDOR_POOL = [
    "Apex Office Interiors", "Bright Harbor Media", "CloudSpan Hosting", "Delta Facility Services",
    "Firstlane Logistics", "Grandview Properties LLC",
    "Helix IT Solutions", "Inkwell Print Co.", "Junction Utility Co-op", "Kite Digital Ads",
    "Lumen Office Supply", "Meridian Legal LLP", "Northgate Equipment Rental",
    "Pillar Accounting Services", "Quicksilver Couriers",
    "Ridgeline Telecom", "Stanton Cleaning Crew", "Trellis Marketing Group",
    "Uptown Catering Co.", "Vertex Materials", "Willow Creek Packaging", "Zenith Freight Lines",
]

# Special-role vendors, seeded per world (kept OUT of VENDOR_POOL so the
# insurance broker / software vendor never collides with rent or driver
# vendors, and so no fixed sentinel name repeats across all worlds).
INSURANCE_VENDORS = [
    "Everline Insurance Brokers", "Hartline Mutual Insurance",
    "Copperfield Risk Partners", "Stately Assurance Group",
    "Bellhaven Insurance Agency", "Crestwell Coverage Co.",
]

SOFTWARE_VENDORS = [
    "Orchard Software Ltd.", "Nimbus Apps Inc.", "Clearstack Software Co.",
    "Bytefield Systems LLC", "Quillware Technologies", "Lattice Desk Software",
]

# Vendor -> expense-account affinity (P3): bills and drivers draw their vendor
# FROM the pool of the account they post to, so vendor identity, description,
# and account are semantically coherent. Keys are account numbers used by the
# ordinary-bill pool and every industry flavor account.
EXPENSE_VENDORS = {
    # Pools are DISJOINT (a vendor appears under exactly one account) so no
    # bill can be read as belonging to a different account than it posts to.
    "6200": ["Bright Harbor Media", "Kite Digital Ads", "Trellis Marketing Group"],
    "6300": ["Stackline SaaS Tools", "Clearbyte Cloud Services"],
    "6350": ["Caldwell Medical Supply", "Meridix Clinical Products",
             "Bluecross Lab Consumables"],
    "6450": ["CloudSpan Hosting", "Helix IT Solutions"],
    "6500": ["Junction Utility Co-op", "Ridgeline Telecom"],
    "6600": ["Meridian Legal LLP", "Pillar Accounting Services"],
    "6700": ["Lumen Office Supply", "Inkwell Print Co.", "Stanton Cleaning Crew",
             "Delta Facility Services", "Uptown Catering Co."],
    "5100": ["Firstlane Logistics", "Quicksilver Couriers", "Zenith Freight Lines",
             "Willow Creek Packaging"],
    "5200": ["Vertex Materials", "Ironvale Tool & Abrasives",
             "Northgate Equipment Rental"],
}

PROPERTY_VENDORS = ["Grandview Properties LLC", "Harborstone Realty Partners",
                    "Amberfield Property Group"]

# One-off variance-driver bill wording, seeded per driver (varies the phrasing
# so no single literal string identifies driver bills across worlds).
DRIVER_DESC_TEMPLATES = [
    "One-time {c} initiative",
    "{c} project - one-off engagement",
    "Non-recurring {c} program",
    "{c} launch sprint (one-time)",
    "Special {c} initiative - one-off fees",
]

# Misposted-bill description variants (seeded per world). Deliberately
# period-expense services (P3a): nothing capitalizable, so the only correct
# treatment is a reclass to the right expense account — never a prepaid.
MISPOST_DESCS = [
    "Trade-show booth build-out and staffing - June expo",
    "Print ad campaign placement - June run",
    "Sponsored newsletter placements for June",
    "June product-launch event promotion services",
]

PEOPLE = ["A. Whitcomb", "B. Okafor", "C. Reyes", "D. Lindqvist", "E. Nakamura",
          "F. Boudreau", "G. Castellano", "H. Adeyemi", "J. Prather", "K. Sorensen",
          "L. Mbeki", "M. Delacroix", "N. Vartanian", "P. Guerrero", "R. Halloway"]

CAMPAIGN_WORDS = ["Lighthouse", "Momentum", "Harvest", "Northstar", "Catalyst",
                  "Bloom", "Velocity", "Anchor", "Summit", "Horizon"]

STREETS = ["Marshfield Rd", "Delancey St", "Whitaker Ave", "Fulton Pike", "Garvey Blvd",
           "Ninth St", "Calloway Dr", "Prospect Ave"]

CITIES = ["Columbus, OH", "Tacoma, WA", "Durham, NC", "Mesa, AZ", "Rockford, IL",
          "Albany, NY", "Chattanooga, TN", "Eugene, OR"]


def pick(rnd, pool, n=1):
    """Deterministically sample n distinct items (order-stable)."""
    items = rnd.sample(list(pool), n)
    return items[0] if n == 1 else items
