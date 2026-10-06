"""The analysis dimensions shared by the analyst implementations and the pipeline.

Each entry is (section_id, title, retrieval_query). The pipeline runs one
retrieval query per section; analysts turn the results into that section of
the report. Keeping the definition here lets the mock and LLM analysts (and
the eval harness) share it without circular imports.
"""

from __future__ import annotations

ANALYSIS_SECTIONS: list[tuple[str, str, str]] = [
    (
        "overview",
        "Site Overview",
        "what is this website about, its purpose and its main offerings",
    ),
    (
        "products",
        "Products & Features",
        "products, features, capabilities and technical details described on the site",
    ),
    (
        "pricing",
        "Pricing",
        "pricing plans, costs, tiers, trials, discounts and billing terms",
    ),
    (
        "audience",
        "Audience & Use Cases",
        "who the products are for, target customers and real-world use cases",
    ),
    (
        "company",
        "Company & Contact",
        "company background, team, contact channels and support options",
    ),
]
