from datasets import load_dataset

# 加载 default、queries、corpus
cache_dir = "/data/share/project/shared_datasets/mteb"
ds_default = load_dataset("mteb/trec-covid", "default", cache_dir=cache_dir)
ds_queries = load_dataset("mteb/trec-covid", "queries", cache_dir=cache_dir)
ds_corpus = load_dataset("mteb/trec-covid", "corpus", cache_dir=cache_dir)


# 建立 _id -> 整行 的映射，便于按 id 查找
queries_list = ds_queries["queries"]
corpus_list = ds_corpus["corpus"]
query_by_id = {row["_id"]: row for row in queries_list}
corpus_by_id = {row["_id"]: row for row in corpus_list}

# 从 test 中筛选 score == 2.0 的文档
test_split = ds_default["test"]
score2_ds = test_split.filter(lambda x: x["score"] == 2.0)
sample_size = min(10, len(score2_ds))
score2_rows = score2_ds.shuffle().select(range(sample_size)).to_dict()

print("=" * 80)
print("Default 数据集 test 集合中 score == 2.0 的 query 与 corpus 具体内容")
print("=" * 80)

total = len(score2_rows["query-id"])
print(f"随机展示 score == 2.0 的样本数量: {total}")

for i in range(total):
    query_id = score2_rows["query-id"][i]
    corpus_id = score2_rows["corpus-id"][i]
    score = score2_rows["score"][i]

    query_row = query_by_id[query_id]
    corpus_row = corpus_by_id[corpus_id]

    print(f"\n【第 {i+1} 条】")
    print(f"  query-id:  {query_id}")
    print(f"  corpus-id: {corpus_id}")
    print(f"  score:     {score}")
    print("-" * 60)
    print("  QUERY 内容:")
    print(f"    {query_row['text']}")
    print("-" * 60)
    print("  CORPUS 内容:")
    title = corpus_row.get("title", "")
    text = corpus_row["text"]
    print(f"    {title} {text}")
    print("=" * 80)


'''
(DSA) root@rjdkbvimokxjpgax-open-d8857fdc8-67skw:/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline# python /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/test_Check_Dataset/test_check_trec-covid.py
================================================================================
Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容
================================================================================

【第 1 条】
  query-id:  17
  corpus-id: fob0os8u
  score:     0.0
------------------------------------------------------------
  QUERY 内容:
    are there any clinical trials available for the coronavirus
------------------------------------------------------------
  CORPUS 内容:
    title: Efficacy of Zinc Against Common Cold Viruses: An Overview
    text:  ABSTRACT Objective To review the laboratory and clinical evidence of the medicinal value of zinc for the treatment of the common cold. Data Sources Published articles identified through Medline (1980–2003) using the search terms zinc, rhinovirus, and other pertinent subject headings. Additional sources were identified from the bibliographies of the retrieved articles. Study Selection By the author. Data Extraction By the author. Data Synthesis Human rhinoviruses, by attaching to the nasal epithelium via the intracellular adhesion molecule-1 (ICAM-1) receptor, cause most colds. Ionic zinc, based on its electrical charge, also has an affinity for ICAM-1 receptor sites and may exert an antiviral effect by attaching to the ICAM-1 receptors in the rhinovirus structure and nasal epithelial cells. Clinical tests of zinc for treatment of common colds have been inconsistent, primarily because of study design, blinding, and lozenge contents. Early formulations of lozenges also were unpalatable. In three trials with similar study designs, methodologies, and efficacy assessments, zinc effectively and significantly shortened the duration of the common cold when it was administered within 24 hours of the onset of symptoms. Recent reports of trials with zinc gluconate administered as a nasal gel have supported these findings; in addition, they have shown that treatment with zinc nasal gel is effective in reducing the duration and severity of common cold symptoms in patients with established illness. Conclusion Clinical trial data support the value of zinc in reducing the duration and severity of symptoms of the common cold when administered within 24 hours of the onset of common cold symptoms. Additional clinical and laboratory evaluations are warranted to further define the role of ionic zinc for the prevention and treatment of the common cold and to elucidate the biochemical mechanisms through which zinc exerts its symptom-relieving effects.
================================================================================

【第 2 条】
  query-id:  31
  corpus-id: p9mbmfaq
  score:     0.0
------------------------------------------------------------
  QUERY 内容:
    How does the coronavirus differ from seasonal flu?
------------------------------------------------------------
  CORPUS 内容:
    title: Evaluation of a Phylogenetic Marker Based on Genomic Segment B of Infectious Bursal Disease Virus: Facilitating a Feasible Incorporation of this Segment to the Molecular Epidemiology Studies for this Viral Agent
    text:  BACKGROUND: Infectious bursal disease (IBD) is a highly contagious and acute viral disease, which has caused high mortality rates in birds and considerable economic losses in different parts of the world for more than two decades and it still represents a considerable threat to poultry. The current study was designed to rigorously measure the reliability of a phylogenetic marker included into segment B. This marker can facilitate molecular epidemiology studies, incorporating this segment of the viral genome, to better explain the links between emergence, spreading and maintenance of the very virulent IBD virus (vvIBDV) strains worldwide. METHODOLOGY/PRINCIPAL FINDINGS: Sequences of the segment B gene from IBDV strains isolated from diverse geographic locations were obtained from the GenBank Database; Cuban sequences were obtained in the current work. A phylogenetic marker named B-marker was assessed by different phylogenetic principles such as saturation of substitution, phylogenetic noise and high consistency. This last parameter is based on the ability of B-marker to reconstruct the same topology as the complete segment B of the viral genome. From the results obtained from B-marker, demographic history for both main lineages of IBDV regarding segment B was performed by Bayesian skyline plot analysis. Phylogenetic analysis for both segments of IBDV genome was also performed, revealing the presence of a natural reassortant strain with segment A from vvIBDV strains and segment B from non-vvIBDV strains within Cuban IBDV population. CONCLUSIONS/SIGNIFICANCE: This study contributes to a better understanding of the emergence of vvIBDV strains, describing molecular epidemiology of IBDV using the state-of-the-art methodology concerning phylogenetic reconstruction. This study also revealed the presence of a novel natural reassorted strain as possible manifest of change in the genetic structure and stability of the vvIBDV strains. Therefore, it highlights the need to obtain information about both genome segments of IBDV for molecular epidemiology studies.
================================================================================

【第 3 条】
  query-id:  8
  corpus-id: xcacty89
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    how has lack of testing availability led to underreporting of true incidence of Covid-19?
------------------------------------------------------------
  CORPUS 内容:
    title: Predict the next moves of COVID-19: reveal the temperate and tropical countries scenario
    text:  The spread of COVID-19 engulfs almost all the countries and territories of the planet, and infections and fatality are increasing rapidly. The first epi-center of its' massive spread was in Wuhan, Hubei province, China having a temperate weather, but the spread has got an unprecedented momentum in European temperate countries mainly in Italy and Spain (as of March 30, 2020). However, Malaysia and Singapore and the neighboring tropical countries of China got relatively low spread and fatality that created a research interest on whether there are potential impacts of weather condition on COVID-19 spread. Adopting the SIR (Susceptible Infected Removed) deviated model to predict potential cases and death in the coming days from COVID-19 was done using the secondary and official sources of data. This study shows that COVID-19 spread and fatality tend to be high across the world but compared to tropical countries, it is going to be incredibly high in the temperate countries having lower temperature (7-16°C) and humidity (80-90%) in last March. However, some literature predicted that this might not to be true, rather irrespective of weather conditions there might be a continuous spread and death. Moreover, a large number of asymptotic COVID-19 carrier in both temperate and tropical countries may re-outbreak in the coming winter. Therefore, a comprehensive global program with the leadership of WHO for testing of entire population of the world is required, which will be very useful for the individual states to take proper political action, social movement and medical services.
================================================================================

【第 4 条】
  query-id:  39
  corpus-id: 8dm6ju5x
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    What is the mechanism of cytokine storm syndrome on the COVID-19?
------------------------------------------------------------
  CORPUS 内容:
    title: Is Hesperidin Essential for Prophylaxis and Treatment of COVID-19 Infection?
    text:  SARS-CoV-2 or COVID-19 is representing the major global burden that implicated more than 4.7 million infected cases and 310 thousand deaths worldwide in less than 6 months. The prevalence of this pandemic disease is expected to rise every day. The challenge is to control its rapid spread meanwhile looking for a specific treatment to improve patient outcomes. Hesperidin is a classical herbal medicine used worldwide for a long time with an excellent safety profile. Hesperidin is a well-known herbal medication used as an antioxidant and anti-inflammatory agent. Available shreds of evidence support the promising use of hesperidin in prophylaxis and treatment of COVID 19. Herein, we discuss the possible prophylactic and treatment mechanisms of hesperidin based on previous and recent findings. Hesperidin can block coronavirus from entering host cells through ACE2 receptors which can prevent the infection. Anti-viral activity of hesperidin might constitute a treatment option for COVID- 19 through improving host cellular immunity against infection and its good anti-inflammatory activity may help in controlling cytokine storm. Hesperidin mixture with diosmin co-administrated with heparin protect against venous thromboembolism which may prevent disease progression. Based on that, hesperidin might be used as a meaningful prophylactic agent and a promising adjuvant treatment option against SARS-CoV-2 infection.
================================================================================

【第 5 条】
  query-id:  35
  corpus-id: e3t1f0rt
  score:     0.0
------------------------------------------------------------
  QUERY 内容:
    What new public datasets are available related to COVID-19?
------------------------------------------------------------
  CORPUS 内容:
    title: Epidemiological Characteristics of COVID-19: A Systemic Review and Meta-Analysis
    text:  Background: Our understanding of the corona virus disease 2019 (COVID-19) continues to evolve. However, there are many unknowns about its epidemiology. Purpose: To synthesize the number of deaths from confirmed COVID-19 cases, incubation period, as well as time from onset of COVID-19 symptoms to first medical visit, ICU admission, recovery and death of COVID-19. Data Sources: MEDLINE, Embase, and Google Scholar from December 01, 2019 through to March 11, 2020 without language restrictions as well as bibliographies of relevant articles. Study Selection: Quantitative studies that recruited people living with or died due to COVID-19. Data Extraction: Two independent reviewers extracted the data. Conflicts were resolved through discussion with a senior author. Data Synthesis: Out of 1675 non-duplicate studies identified, 57 were included. Pooled mean incubation period was 5.84 (99% CI: 4.83, 6.85) days. Pooled mean number of days from the onset of COVID-19 symptoms to first clinical visit was 4.82 (95% CI: 3.48, 6.15), ICU admission was 10.48 (95% CI: 9.80, 11.16), recovery was 17.76 (95% CI: 12.64, 22.87), and until death was 15.93 (95% CI: 13.07, 18.79). Pooled probability of COVID-19-related death was 0.02 (95% CI: 0.02, 0.03). Limitations: Studies are observational and findings are mainly based on studies that recruited patient from clinics and hospitals and so may be biased toward more severe cases. Conclusion: We found that the incubation period and lag between the onset of symptoms and diagnosis of COVID-19 is longer than other respiratory viral infections including MERS and SARS; however, the current policy of 14 days of mandatory quarantine for everyone might be too conservative. Longer quarantine periods might be more justified for extreme cases.
================================================================================
'''

'''
(DSA) root@rjdkbvimokxjpgax-open-d8857fdc8-67skw:/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline# python /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/test_Check_Dataset/test_check_trec-covid.py
================================================================================
Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容
================================================================================

【第 1 条】
  query-id:  30
  corpus-id: yzr7ifbj
  score:     1.0
------------------------------------------------------------
  QUERY 内容:瑞德西韦是治疗新冠肺炎的有效方法吗
    is remdesivir an effective treatment for COVID-19
------------------------------------------------------------
  CORPUS 内容:2019年新型冠状病毒（2019-nCoV，后命名为SARS-CoV-2）在中国武汉突然爆发，并迅速发展为全球大流行。这标志着致病性冠状病毒第三次传入人类社会，不仅影响了医疗系统，也对全球经济造成了冲击。尽管在经历了前两次冠状病毒事件后，我们对冠状病毒的了解有了巨大飞跃，但在治疗和流行病学控制方面仍缺乏有效的方法。

在本文中，我们简要概述了SARS-CoV-2的流行病学、临床特征和分子特征。我们总结了来自武汉早期研究的当前流行病学和临床数据，并着重强调了SARS-CoV-2与SARS-CoV（严重急性呼吸综合征冠状病毒）及MERS-CoV（中东呼吸综合征冠状病毒）的一些不同特征，例如疾病表现的高度变异性。

我们对新冠肺炎大流行爆发后迅速启动的当前临床试验进行了系统整理。虽然基于SARS-CoV-2基因组的特异性疫苗和治疗性抗体试验目前正在进行中，但这种解决方案更偏向长期，因为它们需要对其安全性进行全面测试。另一方面，重新利用先前为其他病毒感染和疾病设计的现有治疗药物，恰好是应对突发大流行的唯一切实可行的快速响应措施，因为这些药物中的大多数已经过安全性测试。

这些药物可大致分为两类：一类是能直接靶向病毒复制周期的药物，另一类是基于免疫疗法的药物，其目的要么是增强先天抗病毒免疫反应，要么是减轻由失调的炎症反应引起的损伤。早期临床研究显示，几种此类药物具有令人期待的治疗潜力，包括法维拉韦（一种干扰病毒复制的广谱抗病毒药物）和羟氯喹（一种重新用途的抗疟药，可干扰病毒的内体进入途径）。

我们推测，当前的大流行紧急情况将触发更多基于大数据分析的系统性药物重定位设计方法。
    title: A Review of SARS-CoV-2 and the Ongoing Clinical Trials
    text:  The sudden outbreak of 2019 novel coronavirus (2019-nCoV, later named SARS-CoV-2) in Wuhan, China, which rapidly grew into a global pandemic, marked the third introduction of a virulent coronavirus into the human society, affecting not only the healthcare system, but also the global economy. Although our understanding of coronaviruses has undergone a huge leap after two precedents, the effective approaches to treatment and epidemiological control are still lacking. In this article, we present a succinct overview of the epidemiology, clinical features, and molecular characteristics of SARS-CoV-2. We summarize the current epidemiological and clinical data from the initial Wuhan studies, and emphasize several features of SARS-CoV-2, which differentiate it from SARS-CoV and Middle East respiratory syndrome coronavirus (MERS-CoV), such as high variability of disease presentation. We systematize the current clinical trials that have been rapidly initiated after the outbreak of COVID-19 pandemic. Whereas the trials on SARS-CoV-2 genome-based specific vaccines and therapeutic antibodies are currently being tested, this solution is more long-term, as they require thorough testing of their safety. On the other hand, the repurposing of the existing therapeutic agents previously designed for other virus infections and pathologies happens to be the only practical approach as a rapid response measure to the emergent pandemic, as most of these agents have already been tested for their safety. These agents can be divided into two broad categories, those that can directly target the virus replication cycle, and those based on immunotherapy approaches either aimed to boost innate antiviral immune responses or alleviate damage induced by dysregulated inflammatory responses. The initial clinical studies revealed the promising therapeutic potential of several of such drugs, including favipiravir, a broad-spectrum antiviral drug that interferes with the viral replication, and hydroxychloroquine, the repurposed antimalarial drug that interferes with the virus endosomal entry pathway. We speculate that the current pandemic emergency will be a trigger for more systematic drug repurposing design approaches based on big data analysis.
================================================================================

【第 2 条】
  query-id:  10
  corpus-id: wkoqqut1
  score:     0.0
------------------------------------------------------------
  QUERY 内容:
    has social distancing had an impact on slowing the spread of COVID-19?
------------------------------------------------------------
  CORPUS 内容:
    title: Covid 19. The paradox of social distancing.
    text:  Before 2020 the term ‘social distancing’ while not new, was barely known. The concept was promoted by the World Health Organisation in 2008 as a public health measure to prevent transmission of influenza, and in various forms it can be identified in reference to epidemics going back hundreds of years. However, in common parlance social distancing is more likely to have been associated with stigma or social class, something with negative connotations, something to be avoided.
================================================================================

【第 3 条】
  query-id:  25
  corpus-id: vpodtbjk
  score:     2.0
------------------------------------------------------------
  QUERY 内容:  哪些生物标志物可以预测2019-nCOV感染的严重临床进程？
    which biomarkers predict the severe clinical course of 2019-nCOV infection?
------------------------------------------------------------
  CORPUS 内容:
    title: A Comprehensive Literature Review on the Clinical Presentation, and Management of the Pandemic Coronavirus Disease 2019 (COVID-19)
    text:  Coronavirus disease 2019 (COVID-19) is a declared global pandemic. There are multiple parameters of the clinical course and management of the COVID-19 that need optimization. A hindrance to this development is the vast amount of misinformation present due to scarcely sourced manuscript preprints and social media. This literature review aims to presents accredited and the most current studies pertaining to the basic sciences of SARS-CoV-2, clinical presentation and disease course of COVID-19, public health interventions, and current epidemiological developments. The review on basic sciences aims to clarify the jargon in virology, describe the virion structure of SARS-CoV-2 and present pertinent details relevant to clinical practice. Another component discussed is the brief history on the series of experiments used to explore the origins and evolution of the phylogeny of the viral genome of SARS-CoV-2. Additionally, the clinical and epidemiological differences between COVID-19 and other infections causing outbreaks (SARS, MERS, H1N1) are elucidated. Emphasis is placed on evidence-based medicine to evaluate the frequency of presentation of various symptoms to create a stratification system of the most important epidemiological risk factors for COVID-19. These can be used to triage and expedite risk assessment. Furthermore, the limitations and statistical strength of the diagnostic tools currently in clinical practice are evaluated. Criteria on rapid screening, discharge from hospital and discontinuation of self-quarantine are clarified. Epidemiological factors influencing the rapid rate of spread of the SARS-CoV-2 virus are described. Accurate information pertinent to improving prevention strategies is also discussed. The penultimate portion of the review aims to explain the involvement of micronutrients such as vitamin C and vitamin D in COVID19 treatment and prophylaxis. Furthermore, the biochemistry of the major candidates for novel therapies is briefly reviewed and a summary of their current status in the clinical trials is presented. Lastly, the current scientific data and status of governing bodies such as the Center of Disease Control (CDC) and the WHO on the usage of controversial therapies such as angiotensin-converting enzyme (ACE) inhibitors, nonsteroidal anti-inflammatory drugs (NSAIDs) (Ibuprofen), and corticosteroids usage in COVID-19 are discussed. The composite collection of accredited studies on each of these subtopics of COVID-19 within this review will enable clarification and focus on the current status and direction in the planning of the management of this global pandemic.
================================================================================

【第 4 条】
  query-id:  46
  corpus-id: uejtwgar
  score:     0.0
------------------------------------------------------------
  QUERY 内容:
    what evidence is there for dexamethasone as a treatment for COVID-19?
------------------------------------------------------------
  CORPUS 内容:
    title: Inhibition of Bruton tyrosine kinase in patients with severe COVID-19
    text:  Patients with severe COVID-19 have a hyperinflammatory immune response suggestive of macrophage activation. Bruton tyrosine kinase (BTK) regulates macrophage signaling and activation. Acalabrutinib, a selective BTK inhibitor, was administered off-label to 19 patients hospitalized with severe COVID-19 (11 on supplemental oxygen; 8 on mechanical ventilation), 18 of whom had increasing oxygen requirements at baseline. Over a 10-14 day treatment course, acalabrutinib improved oxygenation in a majority of patients, often within 1-3 days, and had no discernable toxicity. Measures of inflammation – C-reactive protein and IL-6 – normalized quickly in most patients, as did lymphopenia, in correlation with improved oxygenation. At the end of acalabrutinib treatment, 8/11 (72.7%) patients in the supplemental oxygen cohort had been discharged on room air, and 4/8 (50%) patients in the mechanical ventilation cohort had been successfully extubated, with 2/8 (25%) discharged on room air. Ex vivo analysis revealed significantly elevated BTK activity, as evidenced by autophosphorylation, and increased IL-6 production in blood monocytes from patients with severe COVID-19 compared with blood monocytes from healthy volunteers. These results suggest that targeting excessive host inflammation with a BTK inhibitor is a therapeutic strategy in severe COVID-19 and has led to a confirmatory international prospective randomized controlled clinical trial.
================================================================================

【第 5 条】
  query-id:  31
  corpus-id: zg17f7bd
  score:     1.0
------------------------------------------------------------
  QUERY 内容:冠状病毒与季节性流感有何不同？
    How does the coronavirus differ from seasonal flu?
------------------------------------------------------------
  CORPUS 内容: 摘要 甲型和乙型流感病毒，以及许多不相关的病毒（包括鼻病毒、呼吸道合胞病毒、腺病毒、偏肺病毒和冠状病毒）都具有相同的季节性，因为这些病毒性急性呼吸道感染（vARIs）在冬季比夏季常见得多。不幸的是，早期使用循环“谱系”病毒株的研究似乎导致微生物学家忽视了民间普遍认为病毒性急性呼吸道感染常发生在受冷之后的观点。如今，无可辩驳的证据表明，环境温度下降和宿主受冷会增加病毒性急性呼吸道感染的发病率和严重程度。本综述探讨了四种可能解释这种关联的机制（M1 - M4）：（M1）冬季人群聚集增加可能促进病毒传播；（M2）较低的温度可能增强病毒体在体外的稳定性；（M3）受冷可能增加宿主的易感性；（M4）较低的温度或宿主受冷可能激活潜伏的病毒体。几乎没有证据支持M1或M2，且这两种机制与热带地区的观察结果不符。一些流行病学异常现象，如病毒性急性呼吸道感染在广阔地理区域内反复同时出现、流感疫情迅速终止以及家庭内流感发病率低等，与M4相符，但与M3（其简单形式）不符。M4似乎是季节性的主要驱动因素，但M3可能也发挥着重要作用。
    title: Seasonality and selective trends in viral acute respiratory tract infections
    text:  Abstract Influenza A and B, and many unrelated viruses including rhinovirus, RSV, adenovirus, metapneumovirus and coronavirus share the same seasonality, since these viral acute respiratory tract infections (vARIs) are much more common in winter than summer. Unfortunately, early investigations that used recycled “pedigree” virus strains seem to have led microbiologists to dismiss the common folk belief that vARIs often follow chilling. Today, incontrovertible evidence shows that ambient temperature dips and host chilling increase the incidence and severity of vARIs. This review considers four possible mechanisms, M1 - 4, that can explain this link: (M1) increased crowding in winter may enhance viral transmission; (M2) lower temperatures may increase the stability of virions outside the body; (M3) chilling may increase host susceptibility; (M4) lower temperatures or host chilling may activate dormant virions. There is little evidence for M1 or M2, which are incompatible with tropical observations. Epidemiological anomalies such as the repeated simultaneous arrival of vARIs over wide geographical areas, the rapid cessation of influenza epidemics, and the low attack rate of influenza within families are compatible with M4, but not M3 (in its simple form). M4 seems to be the main driver of seasonality, but M3 may also play an important role.
================================================================================
'''