
----

## FAO: [CFSAM national cereal production estimate](https://www.fao.org/markets-and-trade/publications/detail/en/c/1679419/)

(Cited in Clingendael report)


"FAO, 2024. Special Report 2023: FAO Crop and Food Supply Assessment Mission (CFSAM) to the Sudan, 19 March 2024"
available here: https://reliefweb.int/report/sudan/special-report-2023-fao-crop-and-food-supply-assessment-mission-cfsam-sudan-19-march-2024-enar


<!-- data/raw/available_calories/FAO_CFSAM_cereal_production.csv -->
<!-- cereal_file = base_path / 'data' / 'raw' / 'available_calories' / 'FAO_CFSAM_cereal_production.csv' -->

```
cereal_file = base_path / 'data' / 'raw' / 'available_calories' / 'FAO_CFSAM_cereal_production.csv'
```

----
## [IPC Analysis Acute Food Insecurity April 2024](https://www.ipcinfo.org/ipc-country-analysis/details-map/en/c/1157066/?iso3=SDN)

Note: I'm not sure this is the latest version - there is conflicting data information:
- "Create date: 19.10.2024" within the xlsx file.
- It was downloaded from the above website that lists:
"RELEASE DATE:  27.06.2024"
"VALIDITY PERIOD: 01.04.2024 > 28.02.2025"
- The file has "April-2024" in the filename.

<!-- data/raw/IPC/SD-IPC-Analysis-Acute-Food-Insecurity-April-2024.xlsx -->

----

### To convert from metric tons of sorghum, millet, wheat to kcal:

["Comparative Study of Nutritional Value of Wheat, Maize, Sorghum, Millet, and Fonio: Some Cereals Commonly Consumed in Côte d’Ivoire"](https://eujournal.org/index.php/esj/article/view/13166/13300) European scientific Journal


parameter Wheat Maize Sorghum Millet Fonio
EV (Kcal / 100 g DM) 308.22 ± 14.36 321.79 ± 18.03 308.84 ± 15.13 319.39 ± 17.67 284.72 ± 19.65


@article{jocelyne2020comparative,
  title={Comparative study of nutritional value of wheat, maize, sorghum, millet, and fonio: some cereals commonly consumed in C{\^o}te d’Ivoire},
  author={Jocelyne, Robet Emilie and B{\'e}hiblo, Konan and Ernest, Amoikon Kouakou},
  journal={European Scientific Journal ESJ},
  volume={16},
  number={21},
  pages={118--131},
  year={2020}
}




<!-- data/raw/available_calories/Jocelyne_etal_nutritional_estimates.csv
 -->


```
kcal_conv_file = base_path / 'data' / 'raw' / 'available_calories' / 'Jocelyne_etal_nutritional_estimates.csv'
```


-----


### Estimated population of Sudan


"""
( 1 ) United Nations Population Division. World Population Prospects: 2022 Revision; ( 2 ) Statistical databases and publications from national statistical offices; ( 3 ) Eurostat: Demographic Statistics; ( 4 ) United Nations Statistics Division. Population and Vital Statistics Reprot ( various years ).

Population, total
Total population is based on the de facto definition of population, which counts all residents regardless of legal status or citizenship. The values shown are midyear estimates.
"""

[World Bank Open Data](https://data.worldbank.org/indicator/SP.POP.TOTL)


<!-- data/raw/population/Sudan_pop_WB_API_SP.POP.TOTL_DS2_en_csv_v2_87 --> 


```
wbpop_file = base_path / 'data' / 'processed' / 'WB_population.csv' 
```


<!-- 
( 1 ) United Nations Population Division. World Population Prospects: 2022 Revision; ( 2 ) Statistical databases and publications from national statistical offices; ( 3 ) Eurostat: Demographic Statistics; ( 4 ) United Nations Statistics Division. Population and Vital Statistics Reprot ( various years ).


Population, total
Total population is based on the de facto definition of population, which counts all residents regardless of legal status or citizenship. The values shown are midyear estimates.

ID: SP.POP.TOTL
Source: ( 1 ) United Nations Population Division. World Population Prospects: 2022 Revision; ( 2 ) Statistical databases and publications from national statistical offices; ( 3 ) Eurostat: Demographic Statistics; ( 4 ) United Nations Statistics Division. Population and Vital Statistics Reprot ( various years ).
License:  CC BY-4.0 
Aggregation Method: Sum
Development Relevance: Increases in human population, whether as a result of immigration or more births than deaths, can impact natural resources and social infrastructure. This can place pressure on a country's sustainability. A significant growth in population will negatively impact the availability of land for agricultural production, and will aggravate demand for food, energy, water, social services, and infrastructure. On the other hand, decreasing population size - a result of fewer births than deaths, and people moving out of a country - can impact a government's commitment to maintain services and infrastructure.
General Comments: Relevance to gender indicator: disaggregating the population composition by gender will help a country in projecting its demand for social services on a gender basis.
Limitations and Exceptions: Current population estimates for developing countries that lack (i) reliable recent census data, and (ii) pre- and post-census estimates for countries with census data, are provided by the United Nations Population Division and other agencies. The cohort component method - a standard method for estimating and projecting population - requires fertility, mortality, and net migration data, often collected from sample surveys, which can be small or limited in coverage. Population estimates are from demographic modeling and so are susceptible to biases and errors from shortcomings in both the model and the data. Because future trends cannot be known with certainty, population projections have a wide range of uncertainty.
Long Definition: Total population is based on the de facto definition of population, which counts all residents regardless of legal status or citizenship. The values shown are midyear estimates.
Periodicity: Annual
Statistical Concept and Methodology: Population estimates are usually based on national population censuses, and estimates of fertility, mortality and migration. Errors and undercounting in census occur even in high-income countries. In developing countries errors may be substantial because of limits in the transport, communications, and other resources required to conduct and analyze a full census. The quality and reliability of official demographic data are also affected by public trust in the government, government commitment to full and accurate enumeration, confidentiality and protection against misuse of census data, and census agencies' independence from political influence. Moreover, comparability of population indicators is limited by differences in the concepts, definitions, collection procedures, and estimation methods used by national statistical agencies and other organizations that collect the data. The currentness of a census and the availability of complementary data from surveys or registration systems are objective ways to judge demographic data quality. Some European countries' registration systems offer complete information on population in the absence of a census. The United Nations Statistics Division monitors the completeness of vital registration systems. Some developing countries have made progress over the last 60 years, but others still have deficiencies in civil registration systems. International migration is the only other factor besides birth and death rates that directly determines a country's population change. Estimating migration is difficult. At any time many people are located outside their home country as tourists, workers, or refugees or for other reasons. Standards for the duration and purpose of international moves that qualify as migration vary, and estimates require information on flows into and out of countries that is difficult to collect. Population projections, starting from a base year are projected forward using assumptions of mortality, fertility, and migration by age and sex through 2050, based on the UN Population Division's World Population Prospects database medium variant.
Topic: Health: Population: Structure
 -->



<!-- 
 ---

 (too specific - protein malnutrition problems)
### Protein Energy Malnutrition](https://www.childrenscolorado.org/globalassets/healthcare-professionals/clinical-pathways/protein-energy-malnutrition.pdf) 
Children's Hospital Colorado
(a.k.a. "Failure to Thrive" which is "a stigmatizing and non-specific term")
 -->

