# Data sources — Norway Housing Pressure Tracker (1992–present)

Retrieval approach for Part 1: download raw data + store with timestamps + record metadata. No cleaning/joining yet.

## 1) SSB StatBank — House Price Index (Table 07221)
- Source: Statistics Norway (SSB) StatBank
- Table: 07221 (House Price Index)
- Retrieval method (Part 1): SSB PxWeb API (v2) via HTTP request
- What it represents: House price index time series (dimensions depend on selected query: region/contents/time)

## 2) SSB StatBank — CPI (Table 08981)
- Source: Statistics Norway (SSB) StatBank
- Table: 08981 (Consumer Price Index)
- Retrieval method (Part 1): SSB PxWeb API (v2) via HTTP request
- What it represents: CPI time series (dimensions depend on selected query: contents/time)

## 3) Norges Bank — Policy rate decisions
- Source: Norges Bank (official website/public table)
- Retrieval method (Part 1): HTML table extraction using pandas.read_html (or stable published table if available)
- What it represents: Policy rate decisions over time (dates and rates/decisions)
