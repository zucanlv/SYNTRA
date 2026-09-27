from datasets import load_dataset
import json

# 加载 default、queries、corpus
cache_dir = "/data/share/project/shared_datasets/mteb"
ds_default = load_dataset("mteb/climate-fever", "default", cache_dir=cache_dir)
ds_queries = load_dataset("mteb/climate-fever", "queries", cache_dir=cache_dir)
ds_corpus = load_dataset("mteb/climate-fever", "corpus", cache_dir=cache_dir)

# 建立 _id -> 整行 的映射，便于按 id 查找
queries_list = ds_queries["queries"]
corpus_list = ds_corpus["corpus"]
query_by_id = {row["_id"]: row for row in queries_list}
corpus_by_id = {row["_id"]: row for row in corpus_list}

# 从 test 中随机抽取 5 条
test_split = ds_default["test"]
sampled_ds = test_split.shuffle(seed=42).select(range(5))
first_five = sampled_ds.to_dict()

print("=" * 80)
print("Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容")
print("=" * 80)

# 用于 Few_Shot_Example.py 兼容格式的列表
apps_few_shot_format = []

for i in range(5):
    query_id = first_five["query-id"][i]
    corpus_id = first_five["corpus-id"][i]
    score = first_five["score"][i]

    query_row = query_by_id[query_id]
    corpus_row = corpus_by_id[corpus_id]

    # 收集 Few_Shot_Example 兼容格式：query + Positive(corpus text)
    apps_few_shot_format.append({
        "query": query_row["text"],
        "Positive": corpus_row["text"]
    })

    print(f"\n【第 {i+1} 条】")
    print(f"  query-id:  {query_id}")
    print(f"  corpus-id: {corpus_id}")
    print(f"  score:     {score}")
    print("-" * 60)
    print("  QUERY 内容:")
    print(f"    {query_row['text']}")
    print("-" * 60)
    print("  CORPUS 内容:")
    print(f"    title: {corpus_row.get('title', '')}")
    text = corpus_row["text"]
    print(f"    text:  {text}")
    print("=" * 80)

# 输出 Few_Shot_Example.py 中 "apps" 兼容的 Python 结构（可直接复制到 Few_Shot_Example.py）
print("\n" + "=" * 80)
print("Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py）：")
print("=" * 80)
print('    "climate-fever": ', end="")
print(json.dumps(apps_few_shot_format, indent=4, ensure_ascii=False))
print("=" * 80)


'''
ct/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/test/test_Check_Dataset/test_check_climate-fever.py
================================================================================
Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容
================================================================================

【第 1 条】
  query-id:  2779
  corpus-id: Diesel_particulate_filter
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Europe and Asia  emit most of the soot from burning coal, wood, dung, and diesel in open  fires or without particulate filters in stoves, chimneys, smokestacks,  and exhaust pipes.
------------------------------------------------------------
  CORPUS 内容:
    title: Diesel particulate filter
    text:  A diesel particulate filter ( or DPF ) is a device designed to remove diesel particulate matter or soot from the exhaust gas of a diesel engine .
================================================================================

【第 2 条】
  query-id:  1698
  corpus-id: 21st_century
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Around 97% of climate experts agree that humans are causing global warming.
------------------------------------------------------------
  CORPUS 内容:
    title: 21st century
    text:  The 21st century is the current century of the Anno Domini era , in accordance with the Gregorian calendar . It began on January 1 , 2001 and will end on December 31 , 2100 . It is the first century of the 3rd millennium . It is distinct from the time span known as the 2000s , which began on January 1 , 2000 and will end on December 31 , 2099 .
================================================================================

【第 3 条】
  query-id:  2876
  corpus-id: Merchants_of_Doubt
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Science entitled The Scientific Consensus on Climate Change (Oreskes 2004).
------------------------------------------------------------
  CORPUS 内容:
    title: Merchants of Doubt
    text:  Merchants of Doubt : How a Handful of Scientists Obscured the Truth on Issues from Tobacco Smoke to Global Warming is a 2010 non-fiction book by American historians of science Naomi Oreskes and Erik M. Conway . It identifies parallels between the global warming controversy and earlier controversies over tobacco smoking , acid rain , DDT , and the hole in the ozone layer . Oreskes and Conway write that in each case `` keeping the controversy alive '' by spreading doubt and confusion after a scientific consensus had been reached , was the basic strategy of those opposing action . In particular , they say that Fred Seitz , Fred Singer , and a few other contrarian scientists joined forces with conservative think tanks and private corporations to challenge the scientific consensus on many contemporary issues .   The George C. Marshall Institute and Fred Singer , two of the subjects , have been critical of the book . Other reviewers have been more favorable . One reviewer said that Merchants of Doubt is exhaustively researched and documented , and may be one of the most important books of 2010 . Another reviewer saw the book as his choice for best science book of the year . It was made into a film , Merchants of Doubt , directed by Robert Kenner , released in 2014 .
================================================================================

【第 4 条】
  query-id:  3048
  corpus-id: Carbon_dioxide
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    There is no question whatsoever that the CO2 increase is human-caused.
------------------------------------------------------------
  CORPUS 内容:
    title: Carbon dioxide
    text:  Carbon dioxide ( chemical formula ) is a colorless gas with a density about 60 % higher than that of air ( 1.225 g/L ) that is odorless at normally encountered concentrations . Carbon dioxide consists of a carbon atom covalently double bonded to two oxygen atoms . It occurs naturally in Earth 's atmosphere as a trace gas at a concentration of about 0.04 percent ( 400 ppm ) by volume . Natural sources include volcanoes , hot springs and geysers , and it is freed from carbonate rocks by dissolution in water and acids . Because carbon dioxide is soluble in water , it occurs naturally in groundwater , rivers and lakes , ice caps , glaciers and seawater . It is present in deposits of petroleum and natural gas .   As the source of available carbon in the carbon cycle , atmospheric carbon dioxide is the primary carbon source for life on Earth and its concentration in Earth 's pre-industrial atmosphere since late in the Precambrian has been regulated by photosynthetic organisms and geological phenomena . Plants , algae and cyanobacteria use light energy to photosynthesize carbohydrate from carbon dioxide and water , with oxygen produced as a waste product .   Carbon dioxide is produced by all aerobic organisms when they metabolize carbohydrates and lipids to produce energy by respiration . It is returned to water via the gills of fish and to the air via the lungs of air-breathing land animals , including humans . Carbon dioxide is produced during the processes of decay of organic materials and the fermentation of sugars in bread , beer and winemaking . It is produced by combustion of wood and other organic materials and fossil fuels such as coal , peat , petroleum and natural gas .   It is a versatile industrial material , used , for example , as an inert gas in welding and fire extinguishers , as a pressurizing gas in air guns and oil recovery , as a chemical feedstock and in liquid form as a solvent in decaffeination of coffee and supercritical drying . It is added to drinking water and carbonated beverages including beer and sparkling wine to add effervescence . The frozen solid form of , known as `` dry ice '' is used as a refrigerant and as an abrasive in dry-ice blasting .   Carbon dioxide is the most significant long-lived greenhouse gas in Earth 's atmosphere . Since the Industrial Revolution anthropogenic emissions - primarily from use of fossil fuels and deforestation - have rapidly increased its concentration in the atmosphere , leading to global warming . Carbon dioxide also causes ocean acidification because it dissolves in water to form carbonic acid .
================================================================================

【第 5 条】
  query-id:  2298
  corpus-id: Florida
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    (2012 is now the hottest by a wide margin), but the USA only comprises 2% of the globe.
------------------------------------------------------------
  CORPUS 内容:
    title: Florida
    text:  Florida -LSB- ˈflɒrᵻdə -RSB- ( Spanish for `` land of flowers '' ) is a state located in the southeastern region of the United States . It is bordered to the west by the Gulf of Mexico , to the north by Alabama and Georgia , to the east by the Atlantic Ocean , and to the south by the Straits of Florida and Cuba . Florida is the 22nd-most extensive , the 3rd-most populous , and the 8th-most densely populated of the U.S. states . Jacksonville is the most populous municipality in the state and is the largest city by area in the contiguous United States . The Miami metropolitan area is Florida 's most populous urban area . The city of Tallahassee is the state capital .   A peninsula between the Gulf of Mexico , the Atlantic Ocean , and the Straits of Florida , it has the longest coastline in the contiguous United States , approximately 1350 mi , and is the only state that borders both the Gulf of Mexico and the Atlantic Ocean . Much of the state is at or near sea level and is characterized by sedimentary soil . The climate varies from subtropical in the north to tropical in the south . The American alligator , American crocodile , Florida panther , and manatee can be found in the Everglades National Park .   Since the first European contact was made in 1513 by Spanish explorer Juan Ponce de León -- who named it La Florida ( -LSB- la floˈɾiða -RSB- `` land of flowers '' ) upon landing there in the Easter season , Pascua Florida -- Florida was a challenge for the European colonial powers before it gained statehood in the United States in 1845 . It was a principal location of the Seminole Wars against the Native Americans , and racial segregation after the American Civil War .   Today , Florida is distinctive for its large Cuban expatriate community and high population growth , as well as for its increasing environmental issues . The state 's economy relies mainly on tourism , agriculture , and transportation , which developed in the late 19th century . Florida is also renowned for amusement parks , orange crops , the Kennedy Space Center , and as a popular destination for retirees .   Florida culture is a reflection of influences and multiple inheritance ; Native American , European American , Hispanic and Latino , and African American heritages can be found in the architecture and cuisine . Florida has attracted many writers such as Marjorie Kinnan Rawlings , Ernest Hemingway and Tennessee Williams , and continues to attract celebrities and athletes . It is internationally known for golf , tennis , auto racing and water sports .
================================================================================

================================================================================
Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py 的 "climate-fever" 键）：
================================================================================
    "climate-fever": [
    {
        "query": "Europe and Asia  emit most of the soot from burning coal, wood, dung, and diesel in open  fires or without particulate filters in stoves, chimneys, smokestacks,  and exhaust pipes.",
        "Positive": "A diesel particulate filter ( or DPF ) is a device designed to remove diesel particulate matter or soot from the exhaust gas of a diesel engine ."
    },
    {
        "query": "Around 97% of climate experts agree that humans are causing global warming.",
        "Positive": "The 21st century is the current century of the Anno Domini era , in accordance with the Gregorian calendar . It began on January 1 , 2001 and will end on December 31 , 2100 . It is the first century of the 3rd millennium . It is distinct from the time span known as the 2000s , which began on January 1 , 2000 and will end on December 31 , 2099 ."
    },
    {
        "query": "Science entitled The Scientific Consensus on Climate Change (Oreskes 2004).",
        "Positive": "Merchants of Doubt : How a Handful of Scientists Obscured the Truth on Issues from Tobacco Smoke to Global Warming is a 2010 non-fiction book by American historians of science Naomi Oreskes and Erik M. Conway . It identifies parallels between the global warming controversy and earlier controversies over tobacco smoking , acid rain , DDT , and the hole in the ozone layer . Oreskes and Conway write that in each case `` keeping the controversy alive '' by spreading doubt and confusion after a scientific consensus had been reached , was the basic strategy of those opposing action . In particular , they say that Fred Seitz , Fred Singer , and a few other contrarian scientists joined forces with conservative think tanks and private corporations to challenge the scientific consensus on many contemporary issues .   The George C. Marshall Institute and Fred Singer , two of the subjects , have been critical of the book . Other reviewers have been more favorable . One reviewer said that Merchants of Doubt is exhaustively researched and documented , and may be one of the most important books of 2010 . Another reviewer saw the book as his choice for best science book of the year . It was made into a film , Merchants of Doubt , directed by Robert Kenner , released in 2014 ."
    },
    {
        "query": "There is no question whatsoever that the CO2 increase is human-caused.",
        "Positive": "Carbon dioxide ( chemical formula ) is a colorless gas with a density about 60 % higher than that of air ( 1.225 g/L ) that is odorless at normally encountered concentrations . Carbon dioxide consists of a carbon atom covalently double bonded to two oxygen atoms . It occurs naturally in Earth 's atmosphere as a trace gas at a concentration of about 0.04 percent ( 400 ppm ) by volume . Natural sources include volcanoes , hot springs and geysers , and it is freed from carbonate rocks by dissolution in water and acids . Because carbon dioxide is soluble in water , it occurs naturally in groundwater , rivers and lakes , ice caps , glaciers and seawater . It is present in deposits of petroleum and natural gas .   As the source of available carbon in the carbon cycle , atmospheric carbon dioxide is the primary carbon source for life on Earth and its concentration in Earth 's pre-industrial atmosphere since late in the Precambrian has been regulated by photosynthetic organisms and geological phenomena . Plants , algae and cyanobacteria use light energy to photosynthesize carbohydrate from carbon dioxide and water , with oxygen produced as a waste product .   Carbon dioxide is produced by all aerobic organisms when they metabolize carbohydrates and lipids to produce energy by respiration . It is returned to water via the gills of fish and to the air via the lungs of air-breathing land animals , including humans . Carbon dioxide is produced during the processes of decay of organic materials and the fermentation of sugars in bread , beer and winemaking . It is produced by combustion of wood and other organic materials and fossil fuels such as coal , peat , petroleum and natural gas .   It is a versatile industrial material , used , for example , as an inert gas in welding and fire extinguishers , as a pressurizing gas in air guns and oil recovery , as a chemical feedstock and in liquid form as a solvent in decaffeination of coffee and supercritical drying . It is added to drinking water and carbonated beverages including beer and sparkling wine to add effervescence . The frozen solid form of , known as `` dry ice '' is used as a refrigerant and as an abrasive in dry-ice blasting .   Carbon dioxide is the most significant long-lived greenhouse gas in Earth 's atmosphere . Since the Industrial Revolution anthropogenic emissions - primarily from use of fossil fuels and deforestation - have rapidly increased its concentration in the atmosphere , leading to global warming . Carbon dioxide also causes ocean acidification because it dissolves in water to form carbonic acid ."
    },
    {
        "query": "(2012 is now the hottest by a wide margin), but the USA only comprises 2% of the globe.",
        "Positive": "Florida -LSB- ˈflɒrᵻdə -RSB- ( Spanish for `` land of flowers '' ) is a state located in the southeastern region of the United States . It is bordered to the west by the Gulf of Mexico , to the north by Alabama and Georgia , to the east by the Atlantic Ocean , and to the south by the Straits of Florida and Cuba . Florida is the 22nd-most extensive , the 3rd-most populous , and the 8th-most densely populated of the U.S. states . Jacksonville is the most populous municipality in the state and is the largest city by area in the contiguous United States . The Miami metropolitan area is Florida 's most populous urban area . The city of Tallahassee is the state capital .   A peninsula between the Gulf of Mexico , the Atlantic Ocean , and the Straits of Florida , it has the longest coastline in the contiguous United States , approximately 1350 mi , and is the only state that borders both the Gulf of Mexico and the Atlantic Ocean . Much of the state is at or near sea level and is characterized by sedimentary soil . The climate varies from subtropical in the north to tropical in the south . The American alligator , American crocodile , Florida panther , and manatee can be found in the Everglades National Park .   Since the first European contact was made in 1513 by Spanish explorer Juan Ponce de León -- who named it La Florida ( -LSB- la floˈɾiða -RSB- `` land of flowers '' ) upon landing there in the Easter season , Pascua Florida -- Florida was a challenge for the European colonial powers before it gained statehood in the United States in 1845 . It was a principal location of the Seminole Wars against the Native Americans , and racial segregation after the American Civil War .   Today , Florida is distinctive for its large Cuban expatriate community and high population growth , as well as for its increasing environmental issues . The state 's economy relies mainly on tourism , agriculture , and transportation , which developed in the late 19th century . Florida is also renowned for amusement parks , orange crops , the Kennedy Space Center , and as a popular destination for retirees .   Florida culture is a reflection of influences and multiple inheritance ; Native American , European American , Hispanic and Latino , and African American heritages can be found in the architecture and cuisine . Florida has attracted many writers such as Marjorie Kinnan Rawlings , Ernest Hemingway and Tennessee Williams , and continues to attract celebrities and athletes . It is internationally known for golf , tennis , auto racing and water sports ."
    }
]
================================================================================

'''



'''
[
    {
        "query": "Some, however, bristle at the belief that because floods and storms have always occurred, they should not be linked to climate change”",
        "Positive": "Climate change denial , or global warming denial , is part of the global warming controversy . It involves denial , dismissal , unwarranted doubt or contrarian views which strongly depart from the scientific opinion on climate change , including the extent to which it is caused by humans , its impacts on nature and human society , or the potential of adaptation to global warming by human actions . Some deniers do endorse the term , but others often prefer the term climate change skepticism , although this is a misnomer for those who deny anthropogenic global warming . In effect , the two terms form a continuous , overlapping range of views , and generally have the same characteristics : both reject , to a greater or lesser extent , mainstream scientific opinion on climate change . Climate change denial can also be implicit , when individuals or social groups accept the science but fail to come to terms with it or to translate their acceptance into action . Several social science studies have analyzed these positions as forms of denialism .   Campaigning to undermine public trust in climate science has been described as a `` denial machine '' of industrial , political and ideological interests , supported by conservative media and skeptical bloggers in manufacturing uncertainty about global warming . In the public debate , phrases such as climate skepticism have frequently been used with the same meaning as climate denialism . The labels are contested : those actively challenging climate science commonly describe themselves as `` skeptics '' , but many do not comply with common standards of scientific skepticism and , regardless of evidence , persistently deny the validity of human caused global warming .   Although scientific opinion on climate change is that human activity is extremely likely to be the primary driver of climate change , the politics of global warming have been affected by climate change denial , hindering efforts to prevent climate change and adapt to the warming climate . Those promoting denial commonly use rhetorical tactics to give the appearance of a scientific controversy where there is none .   Of the world 's countries , the climate change denial industry is most powerful in the United States . Since January 2015 , the United States Senate Committee on Environment and Public Works has been chaired by oil lobbyist and climate change denier Jim Inhofe . Inhofe is notorious for having called climate change `` the greatest hoax ever perpetrated against the American people '' and for having claimed to have debunked the alleged hoax in February 2015 when he brought a snowball with him in the Senate chamber and tossed it across the floor . Organised campaigning to undermine public trust in climate science is associated with conservative economic policies and backed by industrial interests opposed to the regulation of emissions . Climate change denial has been associated with the fossil fuels lobby , the Koch brothers , industry advocates and libertarian think tanks , often in the United States . More than 90 % of papers sceptical on climate change originate from right-wing think tanks .  The total annual income of these climate change counter-movement-organizations is roughly $ 900 million . Between 2002 and 2010 , nearly $ 120 million ( # 77 million ) was anonymously donated via the Donors Trust and Donors Capital Fund to more than 100 organisations seeking to undermine the public perception of the science on climate change . In 2013 the Center for Media and Democracy reported that the State Policy Network ( SPN ) , an umbrella group of 64 U.S. think tanks , had been lobbying on behalf of major corporations and conservative donors to oppose climate change regulation .   Since the late 1970s , oil companies have published research broadly in line with the standard views on global warming . Despite this , oil companies organized a climate change denial campaign to disseminate public disinformation for several decades , a strategy that has been compared to the organized denial of the hazards of tobacco smoking by tobacco companies ."
    },
    {
        "query": "We’ll still be facing extreme heat, but at a far more manageable level than if we’d done nothing to halt climate change.",
        "Positive": "Climate change has been a major issue in Australia since the beginning of the 21st century . In 2013 , the CSIRO released a report stating that Australia is becoming hotter , and that it will experience more extreme heat and longer fire seasons because of climate change . In 2014 , the Bureau of Meteorology released a report on the state of Australia 's climate that highlighted several key points , including the significant increase in Australia 's temperatures ( particularly night-time temperatures ) and the increasing frequency of bush fires , droughts and floods , which have all been linked to climate change .   Since the beginning of the 20th century Australia has experienced an increase of nearly 1 ° C in average annual temperatures , with warming occurring at twice the rate over the past 50 years than in the previous 50 years . Recent climate events such as extremely high temperatures and widespread drought have focused government and public attention on the impacts of climate change in Australia . Rainfall in southwestern Australia has decreased by 10 -- 20 % since the 1970s , while southeastern Australia has also experienced a moderate decline since the 1990s . Rainfall patterns are expected to be problematic , as rain has become heavier and infrequent , as well as more common in summer rather than in winter , with little or no uptrend in rainfall in the Western Plateau and the Central Lowlands of Australia . Water sources in the southeastern areas of Australia have depleted due to increasing population in urban areas ( rising demand ) coupled with climate change factors such as persistent prolonged drought ( diminishing supply ) . At the same time , Australia continues to have the highest per capita greenhouse gas emissions . Temperatures in Australia have also risen dramatically since 1910 and nights have become warmer .   A carbon tax was introduced in 2011 by the Gillard government in an effort to reduce the impact of climate change and despite some criticism , it successfully reduced Australia 's carbon dioxide emissions , with coal generation down 11 % since 2008 -- 09 . The subsequent Australian Government , elected in 2013 under then Prime Minister Tony Abbott was criticised for being `` in complete denial about climate change '' . Furthermore , the Abbott government repealed the carbon tax on 17 July 2014 in a heavily criticised move . The renewable energy target ( RET ) , launched in 2001 , was heavily modified under Abbott 's government . However , under the government of Malcolm Turnbull , Australia attended the 2015 United Nations Climate Change Conference and adopted the Paris Agreement . This agreement includes a review of emission reduction targets every 5 years from 2020 .   The federal government and all state governments ( New South Wales , Victoria , Queensland , South Australia , Western Australia , Tasmania , Northern Territory and the Australian Capital Territory ) have explicitly recognised that climate change is being caused by greenhouse gas emissions , in conformity with the scientific opinion on climate change . Sectors of the population have campaigned against new coal mines and coal-fired power stations , reflecting concerns about the effects of global warming on Australia .  The Garnaut Climate Change Review predicted that a net benefit to Australia may be derived by stabilising greenhouse gases in the atmosphere at 450ppm CO2 eq .   The per-capita carbon footprint in Australia was rated 12th in the world by PNAS in 2011 , considerably large given the small population of the country ."
    },
    {
        "query": "The warming is extremely rapid on the geologic time scale, and no other factor can explain it as well as human emissions of greenhouse gases.",
        "Positive": "Carbon dioxide ( chemical formula ) is a colorless gas with a density about 60 % higher than that of air ( 1.225 g/L ) that is odorless at normally encountered concentrations . Carbon dioxide consists of a carbon atom covalently double bonded to two oxygen atoms . It occurs naturally in Earth 's atmosphere as a trace gas at a concentration of about 0.04 percent ( 400 ppm ) by volume . Natural sources include volcanoes , hot springs and geysers , and it is freed from carbonate rocks by dissolution in water and acids . Because carbon dioxide is soluble in water , it occurs naturally in groundwater , rivers and lakes , ice caps , glaciers and seawater . It is present in deposits of petroleum and natural gas .   As the source of available carbon in the carbon cycle , atmospheric carbon dioxide is the primary carbon source for life on Earth and its concentration in Earth 's pre-industrial atmosphere since late in the Precambrian has been regulated by photosynthetic organisms and geological phenomena . Plants , algae and cyanobacteria use light energy to photosynthesize carbohydrate from carbon dioxide and water , with oxygen produced as a waste product .   Carbon dioxide is produced by all aerobic organisms when they metabolize carbohydrates and lipids to produce energy by respiration . It is returned to water via the gills of fish and to the air via the lungs of air-breathing land animals , including humans . Carbon dioxide is produced during the processes of decay of organic materials and the fermentation of sugars in bread , beer and winemaking . It is produced by combustion of wood and other organic materials and fossil fuels such as coal , peat , petroleum and natural gas .   It is a versatile industrial material , used , for example , as an inert gas in welding and fire extinguishers , as a pressurizing gas in air guns and oil recovery , as a chemical feedstock and in liquid form as a solvent in decaffeination of coffee and supercritical drying . It is added to drinking water and carbonated beverages including beer and sparkling wine to add effervescence . The frozen solid form of , known as `` dry ice '' is used as a refrigerant and as an abrasive in dry-ice blasting .   Carbon dioxide is the most significant long-lived greenhouse gas in Earth 's atmosphere . Since the Industrial Revolution anthropogenic emissions - primarily from use of fossil fuels and deforestation - have rapidly increased its concentration in the atmosphere , leading to global warming . Carbon dioxide also causes ocean acidification because it dissolves in water to form carbonic acid ."
    },
    {
        "query": "The claim sea level isn’t rising is based on blatantly doctored graphs contradicted by observations.",
        "Positive": "Climate Change 2007 , the Fourth Assessment Report ( AR4 ) of the United Nations Intergovernmental Panel on Climate Change ( IPCC ) , is the fourth in a series of reports intended to assess scientific , technical and socio-economic information concerning climate change , its potential effects , and options for adaptation and mitigation . The report is the largest and most detailed summary of the climate change situation ever undertaken , produced by thousands of authors , editors , and reviewers from dozens of countries , citing over 6,000 peer-reviewed scientific studies .   It supersedes the Third Assessment Report ( 2001 ) , and is superseded by the Fifth Assessment Report .   The headline findings of the report were : `` warming of the climate system is unequivocal '' , and `` most of the observed increase in global average temperatures since the mid-20th century is very likely due to the observed increase in anthropogenic greenhouse gas concentrations . ''"
    },
    {
        "query": "Water vapour is the most dominant greenhouse gas.",
        "Positive": "Earth's climate arises from the interaction of five major climate system components: the atmosphere (air), the hydrosphere (water), the cryosphere (ice and permafrost), the lithosphere (earth's upper rocky layer) and the biosphere (living things). Climate is the average weather, typically over a period of 30 years, and is determined by a combination of processes in the climate system, such as ocean currents and wind patterns. Circulation in the atmosphere and oceans is primarily driven by solar radiation and transports heat from the tropical regions to regions that receive less energy from the Sun. The water cycle also moves energy throughout the climate system. In addition, different chemical elements, necessary for life, are constantly recycled between the different components. The climate system can change due to internal variability and external forcings. These external forcings can be natural, such as variations in solar intensity and volcanic eruptions, or caused by humans. Accumulation of heat-trapping greenhouse gases, mainly being emitted by people burning fossil fuels, is causing global warming. Human activity also releases cooling aerosols, but their net effect is far less than that of greenhouse gases. Changes can be amplified by feedback processes in the different climate system components."
    }
]

'''