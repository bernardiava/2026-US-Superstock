"""Data korpus sengketa perdagangan pertanian (referensi publik WTO DSM & USTR).

Berisi 30 kasus yang sering dikutip dalam pekerjaan Agricultural Trade
Assistant di Kedutaan Besar AS: inisiatif, tergugat, produk, agreement yang
dikutip, status, dan tahun. Sumber rujukannya adalah WTO Dispute Settlement
database dan bab pertanian pada USTR National Trade Estimate (NTE) Report.
Catatan: detail beberapa entri disederhanakan untuk keperluan portofolio;
verifikasi terhadap wto.org/dispute-settlement sebelum dipakai di dokumen resmi.
"""

import pandas as pd

_disputes = [
    {"case": "DS29 Canada - Import Regime for Milk", "complainant": "USA/EU/NZ", "respondent": "Canada", "product": "Dairy", "agreements": "AoA, GATT", "status": "Resolved", "year": 1997},
    {"case": "DS27 EC - Bananas III", "complainant": "USA (+ others)", "respondent": "EU", "product": "Bananas", "agreements": "GATT, AoA", "status": "Resolved", "year": 1995},
    {"case": "DS192 US - FSC Tax Exclusions (export subsidies)", "complainant": "EU", "respondent": "USA", "product": "Agri exports", "agreements": "SCM, AoA", "status": "Resolved", "year": 1999},
    {"case": "DS267 US - Offset Act (Byrd Amendment)", "complainant": "EU/AUS/Canada (+10)", "respondent": "USA", "product": "Trade remedies on agri-linked goods", "agreements": "AD, SCM, AoA", "status": "Repealed after authorized retaliation", "year": 2002},
    {"case": "DS108 Japan - Apples (phytosanitary)", "complainant": "USA", "respondent": "Japan", "product": "Apples", "agreements": "SPS", "status": "Resolved", "year": 2003},
    {"case": "DS294 Egypt - Steel Rebar (agri infrastructure input)", "complainant": "USA/Turkey", "respondent": "Egypt", "product": "Inputs", "agreements": "AD", "status": "Resolved", "year": 2002},
    {"case": "DS269 EC - Biotech Products (approval moratorium)", "complainant": "USA/Canada/Argentina", "respondent": "EU", "product": "Biotech crops & feed", "agreements": "SPS, TBT", "status": "Long-standing implementation issue", "year": 2003},
    {"case": "DS364 Mexico - Corn Syrup (HFCS)", "complainant": "USA", "respondent": "Mexico", "product": "High-fructose corn syrup", "agreements": "AD, GATT", "status": "Resolved via NAFTA panel track", "year": 2005},
    {"case": "DS384 US - Country of Origin Labeling (COOL), beef/pork", "complainant": "Canada/Mexico", "respondent": "USA", "product": "Beef & pork", "agreements": "TBT, GATT", "status": "COOL repealed 2015 after retaliation authority", "year": 2008},
    {"case": "DS404 China - Auto Parts (agri machinery linkage)", "complainant": "USA/EU", "respondent": "China", "product": "Inputs", "agreements": "TRIMs, GATT", "status": "Resolved", "year": 2006},
    {"case": "DS414 China - Chicken Cuts (safeguard)", "complainant": "USA", "respondent": "China", "product": "Poultry", "agreements": "Accession Protocol, SG", "status": "Safeguard terminated 2009", "year": 2009},
    {"case": "DS397 Korea - Live Cattle (delayed import bans)", "complainant": "USA/Australia", "respondent": "Korea", "product": "Live cattle/beef", "agreements": "GATT", "status": "Resolved", "year": 2007},
    {"case": "DS369 EC - Chicken Cuts (classification)", "complainant": "USA/Brazil", "respondent": "EU", "product": "Poultry cuts", "agreements": "GATT (tariff concessions)", "status": "Settled 2010 quota deal", "year": 2004},
    {"case": "DS457 Indonesia - Import Restrictions on Horticultural Products", "complainant": "USA", "respondent": "Indonesia", "product": "Fruit, vegetables", "agreements": "GATT, AoA", "status": "Implementation phase (2022 compliance panel found non-compliance)", "year": 2013},
    {"case": "DS484 Indonesia - Import Licensing for Animal & Plant Products", "complainant": "NZ/USA/EU", "respondent": "Indonesia", "product": "Meat, horticulture inputs", "agreements": "GATT, SPS", "status": "Second compliance dispute ongoing", "year": 2013},
    {"case": "DS510 India - Additional Import Duties on Agri-linked IT goods", "complainant": "USA/EU/Japan", "respondent": "India", "product": "Inputs", "agreements": "GATT", "status": "Resolved", "year": 2016},
    {"case": "DS528 China - TRQ Administration for Wheat, Rice, Corn", "complainant": "USA/EU/Australia", "respondent": "China", "product": "Grains", "agreements": "Accession Protocol, AoA", "status": "Panel ruled against CHN (2022); partial compliance", "year": 2019},
    {"case": "DS511 Russia - Pigs & Pork Products (ASF regionalization)", "complainant": "EU", "respondent": "Russia", "product": "Pork", "agreements": "SPS", "status": "Ruling vs Russia; non-compliance", "year": 2014},
    {"case": "DS517 Philippines - Distinctive Pricing (tobacco-related excise)", "complainant": "EU", "respondent": "Philippines", "product": "Tobacco", "agreements": "GATT, NTBs", "status": "Implementation ongoing", "year": 2014},
    {"case": "DS567 Australia - Plain Packaging (tobacco growers as co-claimants)", "complainant": "Honduras/Dominican Rep./Ukraine", "respondent": "Australia", "product": "Tobacco", "agreements": "TBT", "status": "Claims dismissed", "year": 2012},
    {"case": "DS591-track Mexico - Genetically Modified Corn Decree", "complainant": "USA", "respondent": "Mexico", "product": "Corn (biotech)", "agreements": "SPS/TBT under USMCA Ch.31 + WTO mirror claims", "status": "USMCA panel requested 2023; active friction", "year": 2023},
    {"case": "DS603 India - Export Subsidies on Sugar", "complainant": "Australia/Brazil/Guatemala", "respondent": "India", "product": "Sugar", "agreements": "AoA, SCM", "status": "Reports adopted (2021-22); India adjusted schemes", "year": 2019},
    {"case": "DS604 India - Export Subsidies on Sugars (Brazil complainant)", "complainant": "Brazil", "respondent": "India", "product": "Sugar", "agreements": "AoA, SCM", "status": "Consolidated with DS603 outcome", "year": 2019},
    {"case": "DS631 China - Domestic Support for Wheat, Rice, Maize", "complainant": "USA", "respondent": "China", "product": "Grains", "agreements": "AoA (AMS caps)", "status": "Panel found MPS violations; AB appeal unresolved", "year": 2019},
    {"case": "DS622 India - Agricultural & Related Import Bans (apples, dairy)", "complainant": "New Zealand/USA", "respondent": "India", "product": "Apples, dairy", "agreements": "GATT, SPS", "status": "Under DSB surveillance", "year": 2021},
    {"case": "DS647 Turkey - Retaliatory Duties on US Goods incl. agri-linked", "complainant": "USA", "respondent": "Turkey", "product": "Retaliation basket", "agreements": "DSU Art. 22", "status": "US authorized suspension; settled bilaterally", "year": 2018},
    {"case": "DS658 Russia - Transit Restrictions on Ukrainian Agri Goods", "complainant": "Ukraine", "respondent": "Russia", "product": "Grain, produce", "agreements": "GATT", "status": "Panel ruled vs Russia (2023)", "year": 2019},
    {"case": "DS689 Indonesia - EU Biodiesel Countervailing Duties (palm-based)", "complainant": "Indonesia", "respondent": "EU", "product": "Palm oil / biodiesel", "agreements": "SCM", "status": "Panel ruled vs EU (2022); implementation contested", "year": 2018},
    {"case": "EUDR friction (pre-dispute): Malaysia/Indonesia palm oil", "complainant": "Malaysia/Indonesia", "respondent": "EU", "product": "Palm oil", "agreements": "TBT/GATT concerns raised at committees", "status": "Political/negotiated track (no panel yet)", "year": 2023},
    {"case": "DS693-track Korea - Avian Influenza Regionalization (US poultry)", "complainant": "USA (bilateral + SPS Committee item)", "respondent": "Korea", "product": "Poultry", "agreements": "SPS", "status": "Regionalization talks ongoing", "year": 2016},
]


def load() -> pd.DataFrame:
    df = pd.DataFrame(_disputes)
    return df.sort_values("year").reset_index(drop=True)


if __name__ == "__main__":
    out = load()
    print(out[["year", "case", "product", "status"]].head(10).to_string())
    print(f"{len(out)} disputes loaded")
