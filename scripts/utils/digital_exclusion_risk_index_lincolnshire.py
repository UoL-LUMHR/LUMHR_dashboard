"""
Digital Exclusion Risk Index (DERI v1.6) Calculation for Lincolnshire 2021 LSOAs.

Methodology: Greater Manchester Combined Authority (GMCA) / Salford City Council.
Source: https://github.com/GreaterManchesterODA/Digital-Exclusion-Risk-Index

Calculates:
1. 0-10 normalized scores for all 10 indicators across England.
2. Demography Component Score (0-10)
3. Deprivation Component Score (0-10)
4. Broadband Component Score (0-10)
5. Overall DERI Score (0-10)
6. Digital Access Score (0-1, where 1 = highest digital access / lowest exclusion risk)
7. Maps 2011 LSOAs to all 435 Lincolnshire 2021 LSOAs via official ONS exact fit lookup.
"""

from pathlib import Path
import json
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

GEOJSON_PATH = PROJECT_ROOT / 'datasets' / 'lsoa' / 'lower-super-output-areas-2021-5RrVTw.geojson'
LOOKUP_PATH = (
    PROJECT_ROOT
    / 'datasets'
    / 'lsoa'
    / 'lsoa_2011_to_2021_lookup'
    / 'LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv'
)
DERI_RAW_PATH = BASE_DIR / 'source_data' / 'digital_exclusion_risk_index' / 'DERI dataset_v1.6.csv'

OUTPUT_LOCAL_CSV = BASE_DIR / 'deri_lincolnshire_2021_lsoa.csv'
OUTPUT_DATASETS_DIR = PROJECT_ROOT / 'datasets' / 'digital_exclusion_risk_index'
OUTPUT_DATASETS_CSV = OUTPUT_DATASETS_DIR / 'deri_lincolnshire_2021_lsoa.csv'


def calculate_deri():
    print(f'Loading raw DERI v1.6 dataset from: {DERI_RAW_PATH}')
    df_raw = pd.read_csv(DERI_RAW_PATH)
    print(f'Total rows in England raw file: {len(df_raw):,}')

    # Indicator columns
    col_age = 'Percentage of population aged 65 and over'
    col_disability = 'Percentage of residents whose day-to-day activities are limited'
    col_no_qual = 'Percentage of residents aged 16+ with no qualifications'
    col_pension = 'Guaranteed pension credit (rate per 1,000 aged 65+)'
    col_unemp = 'Unemployment rate'
    col_social_de = 'Percentage of population in social grade DE'
    col_imd = 'Index of Multiple Deprivation 2019 score - England base'
    col_no_superfast = 'Percentage of homes unable to receive at least 30Mbit/s broadband'
    col_speed = 'Average download speed (Mbit/s)'
    col_slow_conn = 'Percentage of connections receiving less than 10Mbit/s broadband'

    num_cols = [
        col_age,
        col_disability,
        col_no_qual,
        col_pension,
        col_unemp,
        col_social_de,
        col_imd,
        col_no_superfast,
        col_speed,
        col_slow_conn,
    ]

    for col in num_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        df_raw[col] = df_raw[col].fillna(df_raw[col].median())

    def score_col(series: pd.Series, invert: bool = False) -> pd.Series:
        c_min = series.min()
        c_max = series.max()
        if c_max == c_min:
            return pd.Series(0.0, index=series.index)
        norm = (series - c_min) / (c_max - c_min)
        if invert:
            norm = 1.0 - norm
        return norm * 10.0

    print('Computing 0-10 normalized indicator and component scores across England...')

    # 1. Demography Indicators (0-10)
    s_age = score_col(df_raw[col_age])
    s_disability = score_col(df_raw[col_disability])
    df_raw['Demography_Score'] = (s_age + s_disability) / 2.0

    # 2. Deprivation Indicators (0-10)
    s_no_qual = score_col(df_raw[col_no_qual])
    s_pension = score_col(df_raw[col_pension])
    s_unemp = score_col(df_raw[col_unemp])
    s_social_de = score_col(df_raw[col_social_de])
    s_imd = score_col(df_raw[col_imd])
    df_raw['Deprivation_Score'] = (s_no_qual + s_pension + s_unemp + s_social_de + s_imd) / 5.0

    # 3. Broadband Indicators (0-10)
    s_no_superfast = score_col(df_raw[col_no_superfast])
    s_speed = score_col(df_raw[col_speed], invert=True)
    s_slow_conn = score_col(df_raw[col_slow_conn])
    df_raw['Broadband_Score'] = (s_no_superfast + s_speed + s_slow_conn) / 3.0

    # Overall DERI Score (0-10)
    df_raw['DERI_Score'] = (
        df_raw['Demography_Score'] + df_raw['Deprivation_Score'] + df_raw['Broadband_Score']
    ) / 3.0

    # Digital Access Score (0-1) where 1.0 = best access / lowest risk
    df_raw['Digital_Access'] = 1.0 - (df_raw['DERI_Score'] / 10.0)

    # Contextual percentage/rate columns
    df_raw['Aged_65_Plus_Pct'] = df_raw[col_age] * 100.0
    df_raw['Day_To_Day_Limited_Pct'] = df_raw[col_disability] * 100.0
    df_raw['No_Qualifications_Pct'] = df_raw[col_no_qual] * 100.0
    df_raw['Pension_Credit_Rate'] = df_raw[col_pension]
    df_raw['Unemployment_Rate'] = df_raw[col_unemp] * 100.0
    df_raw['Social_Grade_DE_Pct'] = df_raw[col_social_de] * 100.0
    df_raw['IMD_2019_Score'] = df_raw[col_imd]
    df_raw['No_Superfast_Broadband_Pct'] = df_raw[col_no_superfast] * 100.0
    df_raw['Avg_Download_Speed_Mbps'] = df_raw[col_speed]
    df_raw['Slow_Connections_Pct'] = df_raw[col_slow_conn] * 100.0

    # Load Lincolnshire 2021 LSOA GeoJSON
    print(f'Loading Lincolnshire LSOA 2021 codes from GeoJSON: {GEOJSON_PATH}')
    with open(GEOJSON_PATH) as f:
        gj = json.load(f)

    lincs_2021 = []
    for feat in gj['features']:
        props = feat.get('properties', {})
        code = props.get('CODE') or props.get('LSOA_CODE') or props.get('LSOA21CD')
        if code:
            lincs_2021.append(code.strip())

    lincs_df = pd.DataFrame({'LSOA21CD': sorted(list(set(lincs_2021)))})
    print(f'Total 2021 Lincolnshire LSOAs: {len(lincs_df)}')

    # Load 2011 to 2021 Lookup
    print(f'Loading 2011 to 2021 LSOA exact-fit lookup: {LOOKUP_PATH}')
    lookup = pd.read_csv(LOOKUP_PATH)

    merged = lincs_df.merge(lookup[['LSOA11CD', 'LSOA21CD', 'LSOA21NM', 'LAD22NM']], on='LSOA21CD', how='left')

    cols_to_keep = [
        'LSOA code',
        'DERI_Score',
        'Demography_Score',
        'Deprivation_Score',
        'Broadband_Score',
        'Digital_Access',
        'Aged_65_Plus_Pct',
        'Day_To_Day_Limited_Pct',
        'No_Qualifications_Pct',
        'Pension_Credit_Rate',
        'Unemployment_Rate',
        'Social_Grade_DE_Pct',
        'IMD_2019_Score',
        'No_Superfast_Broadband_Pct',
        'Avg_Download_Speed_Mbps',
        'Slow_Connections_Pct',
    ]

    merged = merged.merge(df_raw[cols_to_keep], left_on='LSOA11CD', right_on='LSOA code', how='left')

    # Aggregate to 2021 LSOA (mean across components for any split/merged zones)
    agg_dict = {c: 'mean' for c in cols_to_keep if c != 'LSOA code'}
    agg_dict['LSOA21NM'] = 'first'
    agg_dict['LAD22NM'] = 'first'

    lsoa21_deri = merged.groupby('LSOA21CD').agg(agg_dict).reset_index()

    # Reorder columns
    out_cols = [
        'LSOA21CD',
        'LSOA21NM',
        'LAD22NM',
        'DERI_Score',
        'Digital_Access',
        'Demography_Score',
        'Deprivation_Score',
        'Broadband_Score',
        'Avg_Download_Speed_Mbps',
        'No_Superfast_Broadband_Pct',
        'Slow_Connections_Pct',
        'Aged_65_Plus_Pct',
        'Day_To_Day_Limited_Pct',
        'No_Qualifications_Pct',
        'Pension_Credit_Rate',
        'Unemployment_Rate',
        'Social_Grade_DE_Pct',
        'IMD_2019_Score',
    ]

    lsoa21_deri = lsoa21_deri[out_cols]
    lsoa21_deri = lsoa21_deri.rename(columns={'LSOA21CD': 'LSOA_Code', 'LSOA21NM': 'LSOA_Name', 'LAD22NM': 'District'})

    print(f'Final Lincolnshire DERI dataset shape: {lsoa21_deri.shape}')
    print(f'Null values check: {lsoa21_deri.isna().sum().to_dict()}')

    # Save to CSV
    OUTPUT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    lsoa21_deri.to_csv(OUTPUT_LOCAL_CSV, index=False)
    lsoa21_deri.to_csv(OUTPUT_DATASETS_CSV, index=False)

    print(f'Saved locally to: {OUTPUT_LOCAL_CSV}')
    print(f'Saved to datasets folder: {OUTPUT_DATASETS_CSV}')


if __name__ == '__main__':
    calculate_deri()
