# 🌾 Agri-Trade Dispute & Policy Monitor

Sebuah proyek **scraping + data analysis + dashboard** yang mensimulasikan dua
kompetensi inti posisi *Agricultural Trade Assistant* di Kedutaan Besar AS:

1. **Trade Problem Resolution** — korpus sengketa/friksi perdagangan pertanian
   nyata (WTO DSM + USTR NTE) dengan pipeline kasework: identifikasi barrier →
   engagement (TIFA/SPS Committee) → eskalasi → resolusi & verifikasi.
2. **International Agreements on Agricultural Policy** — arsitektur aturan yang
   mendasari setiap keputusan kasus: WTO Agreement on Agriculture (3 pilar),
   SPS & TFA, USMCA, RCEP, dan Indonesia–EU CEPA (ditandatangani Des 2025).

Kedua tab tersebut eksplisit ada di dashboard sehingga bisa dikutip langsung
di CV/wawancara.

## Struktur

```
agri_trade_monitor/
├── config.py            # negara, indikator World Bank, daftar perjanjian
├── scraper.py           # penarikan data API World Bank -> data/raw + processed
├── dispute_data.py      # korpus 30 sengketa agri (referensi publik WTO/USTR)
├── analysis.py          # korelasi & agregasi + 4 chart PNG ke output/
└── dashboard.py         # Streamlit app (5 tab)
```

## Menjalankan

```bash
pip install -r requirements.txt
python -m agri_trade_monitor.scraper        # tarik data live (butuh internet)
python -m agri_trade_monitor.analysis       # analisis + visualisasi statis
streamlit run agri_trade_monitor/dashboard.py
```

## Data

- **Live scrape**: World Bank Open Data API v2 — 5 indikator × 23 negara
  (2010–2024): agricultural land, trade % of GDP, agriculture % of GDP, cereal
  production, freshwater withdrawals. Tersimpan di `data/raw/` dan
  `data/processed/` agar reproducible tanpa re-scrape.
- **Dispute corpus**: 30 kasus yang disederhanakan dari WTO Dispute Settlement
  database & bab pertanian USTR National Trade Estimate. Verifikasi ulang ke
  wto.org sebelum dikutip dalam dokumen resmi.

## Temuan awal (siap dipakai untuk talking points)

- 22 dari 30 kasus melibatkan AS sebagai inisiator atau tergugat.
- **SPS** adalah agreement paling sering dikutip pada friksi produk segar
  (apel, unggas, pork, biotech) — konsisten dengan pekerjaan daily post
  reporting seorang agri trade assistant.
- Korelasi openness perdagangan (Trade % GDP, World Bank) vs frekuensi
  sengketa per negara: r ≈ 0.13 → keterbukaan saja bukan prediktor; barrier
  non-tarif (SPS/licensing) yang dominan. Ini argumen kuat untuk tab
  *Trade Problem Resolution*.

## Catatan portofolio

Proyek ini menunjukkan: API scraping dengan retry & normalisasi ISO-code,
pandas reshaping (long↔wide), quality summary otomatis, matplotlib + Altair
visualizations, dan desain dashboard berbasis alur kerja kebijakan nyata.
