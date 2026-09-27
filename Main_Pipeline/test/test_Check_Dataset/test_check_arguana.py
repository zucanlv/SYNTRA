"""
从 default 数据集的 test 集合中随机抽取五条，并分别展示每条对应的 query 与 corpus 的完整内容。
"""

from datasets import load_dataset

# 加载 default、queries、corpus
cache_dir = "/data/share/project/shared_datasets/mteb"
ds_default = load_dataset("mteb/arguana", "default", cache_dir=cache_dir)
ds_queries = load_dataset("mteb/arguana", "queries", cache_dir=cache_dir)
ds_corpus = load_dataset("mteb/arguana", "corpus", cache_dir=cache_dir)

# 建立 _id -> 整行 的映射，便于按 id 查找
queries_list = ds_queries["queries"]
corpus_list = ds_corpus["corpus"]
query_by_id = {row["_id"]: row for row in queries_list}
corpus_by_id = {row["_id"]: row for row in corpus_list}

# 从 test 中随机抽取 5 条
test_split = ds_default["test"]
sampled_ds = test_split.shuffle(seed=123).select(range(5))
first_five = sampled_ds.to_dict()

print("=" * 80)
print("Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容")
print("=" * 80)

for i in range(5):
    query_id = first_five["query-id"][i]
    corpus_id = first_five["corpus-id"][i]
    score = first_five["score"][i]

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
    print(f"    title: {corpus_row.get('title', '')}")
    text = corpus_row["text"]
    print(f"    text:  {text}")
    print("=" * 80)

'''
(DSA) root@rjdkbvimokxjpgax-open-d8857fdc8-67skw:/data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline# python /data/share/project/zucan/Synthetic_Data_code/SyntheticDataResearch/Main_Pipeline/test_Check_Dataset/test_check_arguana.py
================================================================================
Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容
================================================================================

【第 1 条】
  query-id:  test-free-speech-debate-ldhwbmclg-pro02a
  corpus-id: test-free-speech-debate-ldhwbmclg-pro02b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Hate speech  The enforcement of the laws proposed in this article will be fraught, complex and difficult. However, the difficulty of administering a law is never a good argument for refusing to enforce it.  The censorship of the written word ended in England with the Lady Chatterley and Oz obscenity trials, but this liberalisation of publication standards has not prevented the state from prosecuting hate speech when it appears in print. It is clear that, although we have more latitude than ever to say or write what we want (no matter how objectionable), standards and taboos continue to exist. We can take it that these taboos are especially important and valuable to the running of a stable society, as they have persisted despite the legal and cultural changes that have taken place over the last fifty years.  Hate speech is prosecuted and censored because of its power to intrude into the lives of individuals who have not consented to receive it. As pointed out in Jeremy Waldron’s response [1] to Timothy Garton Ash’s piece [2] on hate speech, hateful comments are not dangerous because they insight gullible individuals to abandon their inhibitions and engage in race riots. Hate speech is harmful because it recreates- cheaply and in front of a very large audience- an atmosphere in which vulnerable minorities are put in fear of becoming the targets of violence and prejudice. Additionally, hate speech harms by defaming groups, by propagating lies and half-truths about practices and beliefs, with the objective of socially isolating those groups.  Gangsta rap does all of these things, yet legal responses to the publication of songs containing such lyrics as “Rape a pregnant bitch and tell my friends I had a threesome,” have been timid at best. Even if we maintain our liberal approach to taboo breaking forms of expression, we can still link hip hop to many of the harms that hate speech produces.  Gangsta rap gives the impression that African-American and Latin-American neighbourhoods throughout the USA are violent, lawless places. Even if the pronouncements of rappers such as 50 cent and NWA are overblown or fictitious they enforce social division by vividly discouraging people from entering or interacting with poor minority communities. They damage those communities directly by creating a fear of criminality that serves to limit trust and cohesion among individual community members. Finally, violent hip hop is also defamatory. It propagates an image of minority communities that emphasises violence, poverty and nihilism, whilst loudly proclaiming its authenticity. It is completely irrelevant that these images of minority communities are produced by members of those communities.  It is on this basis, however protracted the process of classification must become, that the content of hip hop songs should be assessed and censored. Liberal democracies are prepared to go to great lengths to adjudicate on speech that could potentially promote racial or religious hatred. The same standards should be applied to hip hop music, because it is capable of producing identical harms.  [1] Waldron, J. “The harm of hate speech”. FreeSpeechDebate, 20 March 2012.  [2] Garton-Ash, T. “Living with difference”. FreeSpeechDebate, 22 January 2012.
------------------------------------------------------------
  CORPUS 内容:
    title: living difference house would ban music containing lyrics glorify
    text:  It is usually the task of movie classification organisations such as the MPAA and the British Board of Film Certification to judge whether the content of a film should be cut or altered. In most cases these groups will be politically independent, but may be politically appointed. They will make the decision to cut content based partly on the criteria described above. A movie will only be censored if it contains shocking or offensive images used in a way that suggests that violence is glamorous, entertaining or without consequences.  There is a broad consensus in western liberal democracies on what constitutes a highly shocking or offensive image. For example, in even the most permissive societies, open and public images of sexual intercourse would be considered problematic. Similarly, graphic depictions of violence against vulnerable individuals would be open to wide condemnation. The thing that unifies each of these categories of image is that they can be easily understood and interpreted by the majority of people. Even a casual observer can understand that pornography is pornography. This is part of the reason why some states try to control extreme images – because they are both powerful and emotive, and easy to produce, display and distribute.  However, music and lyrics are different from images. Language contains a degree of abstraction, depth and nuance that only the most unconventional (and non-commercial) film could replicate. This is problematic, because it is much harder for censors and members of the general public to agree on an exact definition of an offensive statement or form of words. Complex legal processes are used to determine whether or not offensive statements are sufficiently offensive to be classed as hate crimes. Even more complex are the legal procedures used to determine when an individual’s reputation has been damaged by allegations published in books or periodicals.  It will be much harder for ratings or certification boards to decide when a particular song is violent or offensive due to the range of meanings and ambiguities that are built into language. For example, the verse  “Got a temper nigga, go ahead, lose your head/ turn your back on me, get clapped and lose your legs/ I walk around gun on my waist, chip on my shoulder/ ‘til I bust a clip in your face, pussy, this beef ain’t over,”  can either be seen as a series of boastful threats, delivered directly by the musician, but it could also be reported speech – a lot of hip hop music is based on narratives or performer’s accounts of past events. It could also be intended to invite condemnation of the behaviour of the character that the speaker has assumed. Hip hop artists frequently use alternative personas and “casts” of characters to add depth to the narrative dimension of their tracks.  Under these circumstances, the process of classifying and censoring potentially violent lyrics is likely to become laborious. More important than the expense that this process will entail is the possibility that the chilling effect of a prolonged classification process will cause music publishers to stop promoting hip hop, metal and other genres linked with violent imagery. Lack of funds will curtail innovation and diversity in these genres.
================================================================================

【第 2 条】
  query-id:  test-free-speech-debate-ldhwbmclg-con03a
  corpus-id: test-free-speech-debate-ldhwbmclg-con03b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    A ban will further marginalise young members of impoverished communities  Hip hop is an extremely diverse musical genre. Surprisingly, this diversity has evolved from highly minimal series of musical principles. At its most basic, raping consists of nothing more than rhyming verses that are delivered to a beat. This simplicity reflects the economically marginalised communities that hip hop emerged from. All that anyone requires in order to learn how to rap, or to participate in hip hop culture, is a pen, some paper and possibly a disc of breaks – the looped drum and bass lines that are used to time rap verses. Thanks to its highly social aspect, hip hop continues to function as an accessible form of creative expression for members of some of impoverished communities in both the west and elsewhere in the world.  Point 7 suggests that free speech flourishes when we respect believers but are not forced to respect their beliefs. Free Speech Debate discusses this principle in the light of religious belief and religious expression. However, it is also relevant when we consider how our appraisal of an individual’s background, culture and values affects our willingness to accept or dismiss what she says.  The positive case for banning- or at least condemning- hip hop often rests on its ability to reinforce the negative stereotypes of impoverished and marginalised communities that are propagated by majority communities. Critics of hip hop note that black men have often been stigmatised as violent, uncivilised and predatory. They claim that many hip hop artists cultivate a purposefully brutal and misogynist persona. The popularity of hip hop reflects the acceptance of this stereotype, and further entrenches discrimination against young black men. This line of thinking portrays hip hop artists as betrayers or exploiters of their communities, reinforcing damaging stereotypes and convincing adolescents that a violent rejection of mainstream society is a way to achieve material success.  Arguments of this type fail to recognise the depth of nuance and meaning that words and word-play can convey. They are predicated on an assumption that the consumers of hip hop engage with it in a simplistic and uncritical way. In short, such arguments see hip hop fans as being simple minded and easily influenced. This perspective neglects the “recognition respect”, the recognition of equality and inherent dignity that is owed to all contributors of a debate. Moreover, it also bars us from properly assessing the “appraisal respect” owed to the content of hip hop and other controversial musical genres. When hip hop is seen as being inherently harmful, and as being targeted at an especially impressionable and vulnerable part of society, we both demean members of that group and prevent robust discussion of rap lyrics themselves. Academics such as John McWhorter see only the advocacy of violence and nihilism in lyrics such as  “You grow in the ghetto, living second rate/ and your eyes will sing a song of deep hate”.  But these are words that can also be interpreted as astute observation on the brutality that is bred by social exclusion. In point of fact, there is little in the previous verse, or those that follow it,  “You’ll admire all the numberbook takers/ thugs, pimps and pushers, and the big money makers”,  that could be interpreted as permitting, popularising or endorsing violence. That is, unless the individual reading the verse had already concluded that its intended audience lacked his own critical perspective and understanding of social norms and values.  Even if an observer were ultimately conclude that a particular hip hop track had no redeeming value, a broad interpretation of point 7 suggests that he should, at the very least, credit its artists and listeners with a modicum of intelligence and reflectiveness. When we approach music with a custodial mind-set, determined to protect young listeners from what we see as harm or exploitation, we prevent those individuals from access a form of speech that may be the only affordable method of expression open to them.  Just as we allow individuals the right to be heard in a language of their choosing (see point 1), we should also accept that perspectives from marginalised communities may not appear in a conventional form. Under these circumstances, it would be dangerous for us to curtail and marginalise a form of speech geared toward discussing the problems faced by impoverished young people that has, against the odds, penetrated the mainstream. We are likely to deepen existing prejudices by viewing rappers and their fans as infantile, impressionable and in need of protection.
------------------------------------------------------------
  CORPUS 内容:
    title: living difference house would ban music containing lyrics glorify
    text:  This argument makes a claim of bias against academics and commentators who portray the audiences that hip hop music is targeted at as vulnerable. Unfortunately, this is a viewpoint that is closer to the truth than the aspirational narrative provided in the opposition side’s case.  Hip hop emerged from environments that were extremely poor and that had been pushed to the margins of society. This situation has persisted until well into this century. The cyclical effects of racism and discrimination continue to be felt in minority communities. Although anti-discrimination laws now protect access to employment and government services, inequalities in cultural capital and high-impact policing have led to the exclusion of large numbers of young men from the social economic opportunities that are made available to middle class society. Under these circumstances, it is entirely appropriate to describe the adolescent inhabitants of impoverished urban communities as vulnerable.  Poverty- either financial or of opportunity- breeds desperation. An individual placed in a situation of urgent need will not have the ability to reason clearly. This is especially true of young people undergoing the difficult transition to adulthood. Adolescence is characterised by a desire to test the boundaries of social norms and parental authority. Therefore, expression that legitimatises and encourages ever more dangerous forms of rebellion should be kept out of the hands of young people. They are unusually susceptible to the behavioural distortions that side opposition goes out of its way to deny.  We limit the content of the media that children and young people can consume all the time, recognising that the process of education and socialisation changes the individual’s relationship to wider society and their ability to which forms of behaviour will best help them to live freely and happily.  Children and teenagers are more impressionable than adults. Similarly, the rate at which individuals mature and develop can vary wildly. We recognise that, for example, exposure to pornography or violent cinema could have serious behaviour consequences for young children. Objections to the restricted availability of pornography are nonsensical, given that they do a great deal to protect children, and present only a minor inconvenience to an adult’s attempts to access such material.  Although we do not place onerous restrictions on the ability of adults to access media of this type, we can be strict in regulating children’s access. This does not constitute a permanent form of censorship, but instead fulfils the broad remit that the state is granted to protect its citizens. Moreover, classification of expression that is geared toward protecting the vulnerable also aids in protecting the primacy and utility of free speech itself. Free expression- as has been restated throughout this exchange- can harm as easily as it liberates. In some instances, the state must temporarily restrict the access of certain classes of people to certain forms of free expression, in order to ensure that free, frank and controversial discussion and expression can take place in society in general.
================================================================================

【第 3 条】
  query-id:  test-philosophy-apessghwba-con03a
  corpus-id: test-philosophy-apessghwba-con03b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Animal research is necessary for the development of truly novel substances  Undoubtedly then, the most beneficial research to mankind is the development of truly novel drugs. Even according to the proposition this represents about a quarter of all new drugs released, which could be seen as significant given the great potential to relieve the suffering beyond our current capacity that such drugs promise.  After the effects, side effects and more complex interactions of a drug have been confirmed using animal and non-animal testing, it will usually pass to what is called a phase I clinical trial - tests on human volunteers to confirm how the drug will interact with human physiology and what dosages it should be administered in. The risk of a human volunteer involved in a phase I trial being harmed is extremely small, but only because animal tests, along with non-animal screening methods are a highly effective way of ensuring that dangerous novel drugs are not administered to humans. In the United Kingdom, over the past twenty years or more, there have been no human deaths as a result of phase I clinical trials.  Novel compounds (as opposed to so-called "me-too" drugs, that make slight changes to an existing treatment) are the substances that hold the most promise for improving human lives and treating previously incurable conditions. However, their novelty is also the reason why it is difficult for scientists to predict whether they may cause harm to humans.  Research into novel compounds would not be possible without either animal testing, or tremendous risk to human subjects, with inevitable suffering and death on the part of the trial volunteers on some occasions. It is difficult to believe that in such circumstances anyone would volunteer, and that even if they did, pharmaceutical companies would be willing to risk the potential legal consequences of administering a substance to them they knew relatively little about. In short, development of novel drugs requires animal experimentation, and would be impossible under the proposition's policy.
------------------------------------------------------------
  CORPUS 内容:
    title: animals philosophy ethics science science general house would ban animal
    text:  This again highlights some of the problems with animal research. In the UK example cited, animal testing had been done, and the dose given to the human volunteers was a tiny fraction of the dose shown to be safe in primates. Animal research is an unreliable indicator of how drugs will react in the human body, and as such alternatives should be sought and improved upon.
================================================================================

【第 4 条】
  query-id:  test-economy-thhghwhwift-con03a
  corpus-id: test-economy-thhghwhwift-con03b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    It hits the most vulnerable part of society hardest  The practical consequence of an additional tax on what the government considers fatty unhealthy food will disproportionately affect the poorest part of the population, who often turn to such food due to economic constraints.  These were the concerns that stopped the Romanian government from introducing a fat tax in 2010. Experts there argued, that the countries people keep turning to junk food simply because they are poor and cannot afford the more expensive fresh produce. What such a fat tax would do is eliminate a very important source of calories from the society’s economic reach and replace the current diet with an even more nutritionally unbalanced one. Even the WHO described such policies as “regressive from an equity perspective.” [1]  Clearly, the government should be focusing its efforts on making healthy fresh produce more accessible and not on making food in general, regardless if it’s considered healthy or not, less accessible for the most vulnerable in our society.  [1] Stracansky, P., 'Fat Tax' May Hurt Poor, published 8/8/2011,  , accessed 9/12/2011
------------------------------------------------------------
  CORPUS 内容:
    title: tax health health general healthcare weight house would implement fat tax
    text:  Even if this policy might cause some families to spend more on their food – even more than they feel like they can afford – it still is more important to start significantly dealing with the obesity epidemic. We feel that nothing short of forcing these low income families – which are also the ones where obesity is most prevalent – to finally change their eating habits will make a dent in the current trend.  But there is a silver lining here. These are also the families that are afflicted most by obesity related diseases. Thus spending a couple dollars more on food now will – necessarily – save them tens of thousands in the form of medical bills.  Reducing obesity will also make them more productive at work and reduce their absenteeism, again offsetting the costs of this tax. [1]  We should look at this tax as a form of paying it forward – spending a little time and effort now and reap the benefits for the individual and the society in the future.  [1] ACOEM, Obesity Linked To Reduced Productivity At Work, published 1/9/2008,  , accessed 9/14/2011
================================================================================

【第 5 条】
  query-id:  test-environment-aiahwagit-pro03a
  corpus-id: test-environment-aiahwagit-pro03b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Endangered animals are a source of pride for African countries  Endangered animals warrant a tougher degree of protection in Africa as they have notable cultural significance. Some groups believe that African elephants have mystic powers attached to them and have coveted them for centuries. [1] African lions have been depicted on the coat of arms for states and institutions both past and present. [2] They are intrinsically linked with Africa’s past and its identity. The extinction of these animals, therefore, would have a negative cultural impact and should be prevented.  [1] University of California, Los Angeles, ‘Elephant: The Animal and its Ivory in African Culture’  [2] Coleman, Q. ‘The importance of African lions’
------------------------------------------------------------
  CORPUS 内容:
    title: animals international africa house would african government implement tougher
    text:  Not all endangered animals have such cultural significance within Africa. Pangolins are armoured mammals which are native to Africa and Asia. Like rhinoceros, pangolins are endangered due to their demand in East Asia. They are relatively unknown however, and therefore have little cultural significance. [1] This is the case for many of Africa’s lesser known endangered species. Any extension of protection for endangered animals based on their cultural significance would be unlikely to save many of these species.  [1] Conniff, R. ‘Poaching Pangolins: An Obscure Creature Faces Uncertain Future’
================================================================================

'''

'''

================================================================================
Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容
================================================================================

【第 1 条】
  query-id:  test-international-aegmeppghw-pro04a
  corpus-id: test-international-aegmeppghw-pro04b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Any country that fulfils the accession criteria should be allowed to join  Turkey was promised a chance to join the EU by a unanimous vote at the Helsinki summit in 1999, when its candidacy was unanimously accepted after three decades of consistent Turkish requests. As a candidate country Turkey should be allowed in once it meets the membership criteria which were first set out in the Copenhagen European Council of 1993. These were stability of institutions guaranteeing democracy, the rule of law, human rights and respect for and protection of minorities, the existence of a functioning market economy as well as the capacity to cope with competitive pressure and market forces within the Union and the ability to take on the obligations of membership including adherence to the aims of political, economic &amp; monetary union. [1] Clearly economic and political reforms are necessary, but that is true of all states attempting to join the EU and should not be used as an excuse to backtrack now. It would be hypocritical to apply one set of criteria to Central and Eastern European states and another to Turkey. Such blatant hypocrisy would have consequences, if the EU is seen to break its promise to Turkey it may turn a potential friend and partner into a suspicious and hostile neighbour.  [1]  European Commission Enlargement, Accession criteria, 30th October 2010
------------------------------------------------------------
  CORPUS 内容:
    title: americas europe global middle east politics politics general house would
    text:  Turkey first applied to join the EU back in the 1960s but there is no document where EU leaders have promised unconditionally to include Turkey in the future. In a decade of candidacy Turkey has managed to satisfy less than half of the chapters, and these are only the minimum prerequisites. Even if they had, past declarations (as opposed to treaties) cannot be held to bind today’s leaders in weighing both their own national interest and the wider European interest. The possibility is therefore a long way off. The possible negative impact of Turkish EU membership upon existing members must be considered. The recent rise of far-right anti-immigration politicians, such as Marine Le Pen, Jorg Haidar and Pym Fortuyn, point to a dangerous public reaction to more open borders and unchecked migration.
================================================================================

【第 2 条】
  query-id:  test-politics-oapdhwinkp-con02a
  corpus-id: test-politics-oapdhwinkp-con02b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    North Korea is an unresolved conflict it can’t simply be ignored  Even if the provocations are sometimes relatively small and ineffective, such as the failed missile launch in April 2012, as a conflict zone they cant simply be ignored by anyone even if they themselves are unlikely to be drawn into any potential conflict. After Rwanda the United Nations promised never again would it allow genocide; [1] how much worse would it be to ignore something that could be a spark to a conflict that could cost millions of lives when we already know there is the potential. The United Nations was created “To maintain international peace and security, and to that end: to take effective collective measures for the prevention and removal of threats to the peace… to bring about … settlement of international disputes or situations which might lead to a breach of the peace” [2] therefore all nations should be attempting to resolve this frozen conflict that could so easily become a shooting war. Wars in Korea have in the past drawn in all the surrounding powers; the Imjin war involved China and Japan, China and Japan again fought over Korea in 1894-5, and the Korean War 1950-53 brought in both the USA and China while Russia and Japan were both involved as supply bases. Clearly the possibility of conflict is not something any power with a stake in Northeast Asia can simply ignore.  It is essential that there is a reaction to every incident just in case that is the incident that spins out of control.  [1] Power, Samantha, ‘Remember the Blood Frenzy of Rwanda’, Los Angeles Times, 4 April 2004,   [2] ‘Article 1 The Purposes of the United Nations are:’, United Nations, 26 June 1945,
------------------------------------------------------------
  CORPUS 内容:
    title: onal asia politics defence house would ignore north korean provocations
    text:  While the United Nations is about creating peace that does not mean that it needs to keep trying the same failed formula. It is clear that multilateral discussions and sanctions have not succeeded in creating positive change in relation to North Korea. Trying new tactics does not mean giving up on the goal of international peace and security.
================================================================================

【第 3 条】
  query-id:  test-politics-glghssi-pro01a
  corpus-id: test-politics-glghssi-pro01b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Scotland has a proud history and has demonstrated since devolution different political interests  The Union has now passed its 300th birthday and throughout that time Scotland has maintained as distinct role and identity. This is grounded in a tradition and history that is quite different from that south of the Border and includes legal and education systems that have always been separate.  That has manifested itself in a distinct policy agenda since devolution and areas such as free care for the elderly and the abolition of student tuition fees.  Despite the opinions of doomsayers before devolution it has been proved as a remarkable success and massive approval throughout the UK with 70% saying it has been a success. [i]  [i] The Scotsman. “70% of Britons support devolution for Scotland, poll suggests” 8 May 2009.
------------------------------------------------------------
  CORPUS 内容:
    title: government local government house supports scottish independence
    text:  There are many differences between devolution and independence. Surviving events such as the banking crisis and the European sovereign debt crisis are much easier within the confines of a larger, richer state such as the UK.  Nobody denies that devolution has, broadly speaking, been a success. However, it’s been achieved in quite a different context than that facing a nation state.  It has left difficult decisions to Westminster. It allows the Scottish Executive the luxury of being oppositionist on issues such as nuclear power, fantasists on renewables while leaving the problem of how to keep the lights on to politicians at Westminster.
================================================================================

【第 4 条】
  query-id:  test-education-pshhghwpba0-con03a
  corpus-id: test-education-pshhghwpba0-con03b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    A school breakfast for all is a greater cost on schools  Everything costs. Providing free school to all breakfasts will cost the government money for ingredients, cafeteria staff, administration, even possibly new facilities. In the USA the Breakfast Program costs $3.3 billion to provide free or reduced price breakfasts to 10.1 million students. [1] There is a limited total amount of money so the cost will mean there is something else the government will not be able to do. This proposal may mean, for example, that the government cannot afford to hire more teachers to reduce class sizes.  [1] Food and Nutrition Service, ‘The School Breakfast Program’, September 2013
------------------------------------------------------------
  CORPUS 内容:
    title: primary secondary health health general house would provide breakfast all 0
    text:  The upfront cost will be paid back. In the future there will be less health care costs. And there will be a more highly educated and skilled population which will mean more economic growth and tax for the government.
================================================================================

【第 5 条】
  query-id:  test-society-simhbrasnba-con01a
  corpus-id: test-society-simhbrasnba-con01b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    We have a duty to help the persecuted  The principles which underlie the asylum regime are as valid as ever. Millions still face persecution, death and torture globally because of who they are or because of their convictions. Democratic countries still have a moral obligation to offer protection to these people. We all recognise it as a horrendous failing by the countries who turned away Jewish refugees in the early days of Nazism where both the United States and the UK turned away large numbers or refugees, [1] and only the Dominican Republic was willing to take in large numbers. [2] This should never happen again. Developed nations have both the wealth and security to make them the best destinations for those seeking refuge.  [1] Perl, William R., ‘The Holocaust conspiracy: an international policy of genocide’, 1989, pp.37-51  [2] Museum of Jewish Heritage, ‘”A Community Born in Pain and Nurtured in Love” Jews who were given refuge by Dominican Republic’, 8 January 2008.
------------------------------------------------------------
  CORPUS 内容:
    title: society immigration minorities house believes right asylum should not be absolute
    text:  It would be nice to offer safety to everyone who genuinely deserved it, but practically it is almost impossible to tell who is genuinely fleeing persecution, and who is simply seeking economic benefit. In many cases there may be a combination of the two. Tracking down the histories of applicants to verify their claim is frequently impossible, and enormously expensive. The point of moral obligations as opposed to legal obligations is that it is the donor who decides how great their sacrifice should be. States may perfectly fairly decide to try to protect refugees in more affordable and uncontroversial ways, such as providing aid to refugee camps and foreign governments who work nearer crisis areas. Accepting refugees is not obligatory.
================================================================================
'''

'''
================================================================================
Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容
================================================================================

【第 1 条】
  query-id:  test-religion-cmrsgfhbr-con01a
  corpus-id: test-religion-cmrsgfhbr-con01b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    This is a victory for democracy – a precious Filipino value - clear majorities in both houses and in the wider public support it  Opposition have conveniently glossed over one critical issue in this debate – that the RH Bill has significant popular support [i] . It also, as has been demonstrated that a majority of elected representatives support it. In itself these two facts provide evidence that modern Filipinos are sick of the fact that around half of the 3.4 million pregnancies each year are unplanned or the atrocious reality that 90,000 women a year seek the help of back street abortionists. When many of these go wrong, they were denied access to medical care and around 1,000 die each year as a result [ii] .  The values for the respect for the life of the mother, the value of life of the child, respect for the opinions of the majority, respect for democracy and placing the future of individuals and society above the outdated mythology of the Church would seem to be alive and well in the decision to pass this bill.  [i] Rauhala, Emily, ‘Culture Wars: After a decade of debate, the Philippines passes Reproductive Health Bill’, Time, 17 December 2012.   [ii] Ibid.
------------------------------------------------------------
  CORPUS 内容:
    title: church marriage religions society gender family house believes reproductive
    text:  Opposition have conveniently glossed over one critical issue in this debate – that the RH Bill has significant popular support [i] . It also, as has been demonstrated that a majority of elected representatives support it. In itself these two facts provide evidence that modern Filipinos are sick of the fact that around half of the 3.4 million pregnancies each year are unplanned or the atrocious reality that 90,000 women a year seek the help of back street abortionists. When many of these go wrong, they were denied access to medical care and around 1,000 die each year as a result [ii] .  The values for the respect for the life of the mother, the value of life of the child, respect for the opinions of the majority, respect for democracy and placing the future of individuals and society above the outdated mythology of the Church would seem to be alive and well in the decision to pass this bill.  [i] Rauhala, Emily, ‘Culture Wars: After a decade of debate, the Philippines passes Reproductive Health Bill’, Time, 17 December 2012.   [ii] Ibid.
================================================================================

【第 2 条】
  query-id:  test-culture-mthbah-pro02a
  corpus-id: test-culture-mthbah-pro02b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    People are given too much choice, which makes them less happy.  Advertising leads to many people being overwhelmed by the endless need to decide between competing demands on their attention – this is known as the tyranny of choice or choice overload. Recent research suggests that people are on average less happy than they were 30 years ago - despite being better off and having much more choice of things to spend their money on1. The claims of adverts crowd in on people, raising expectations about a product and leading to inevitable disappointment after it is bought. A recent advertisement for make-up was banned in Britain due to the company presenting its product as being more effective than it actually was2. Shoppers feel that a poor purchase is their fault for not choosing more wisely, and regret not choosing something else instead. Some people are so overwhelmed that they cannot choose at all.  1Schwartz, The Tyranny of Choice, 2004.  2 Kekeh , Too Beautiful? British MP Draws Line in Sand for Cosmetic Ads , 2011.
------------------------------------------------------------
  CORPUS 内容:
    title: media television house believes advertising harmful
    text:  People are unhappy because they can't have everything, not because they are given too much choice and find it stressful. In fact, advertisements play a crucial role in ensuring that what money people have, they spend on the most appropriate product for themselves. If advertisements were not permitted, people would waste money on an initial product when, given the choice, they clearly would go for another.  A meta-analysis incorporating research from 50 independent studies found no meaningful connection between choice and anxiety, but speculated that the variance in the studies left open the possibility that choice overload could be tied to certain highly specific and as yet poorly understood pre-conditions1.  1 ^ Scheibehenne, Benjamin; Greifeneder, R. &amp; Todd, P. M. (2010). "Can There Ever be Too Many Options? A Meta-Analytic Review of Choice Overload" . Journal of Consumer Research 37: 409-425.
================================================================================

【第 3 条】
  query-id:  test-health-hdond-con04a
  corpus-id: test-health-hdond-con04b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    People may have valid religious reasons not to donate organs  Many major religions, such as some forms of Orthodox Judaism {Haredim Issue}, specifically mandate leaving the body intact after death. To create a system that aims to strongly pressure people, with the threat of reduced priority for life-saving treatment, to violate their religious beliefs violates religious freedom. This policy would put individuals and families in the untenable position of having to choose between contravene the edicts of their god and losing the life of themselves or a loved one. While it could be said that any religion that bans organ donation would presumably ban receiving organs as transplants, this is not actually the case; some followers of Shintoism and Roma faiths prohibit removing organs from the body, but allow transplants to the body.
------------------------------------------------------------
  CORPUS 内容:
    title: healthcare deny organs non donors
    text:  In reality, the majority of faiths that ban organ donation, and all of the faiths that feel particularly strongly about it, such as certain branches of the Jehovah’s Witness with regard to blood transfusions {Blood – Vital for Life}, also ban accepting foreign organs. In such cases, practitioners wouldn’t be receiving organs anyway, so the net effect is nil. Moreover, many religions mandate that followers do everything in their power to save a life, and that this should trump adherence to lesser dictates. Finally, to adhere to a religious ban on giving but not receiving organs is disingenuous. It is the ultimate hypocrisy: to rely on others to do someone one would not do oneself. In such a situation, the state is no longer obliged to guarantee a chance to adhere to one’s religion.
================================================================================

【第 4 条】
  query-id:  test-international-sepiahbaaw-pro02a
  corpus-id: test-international-sepiahbaaw-pro02b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Environmental Damage  Both licit and illicit resource extraction have caused ecological and environmental damage in Africa. The procurement of many natural resources requires processes such as mining and deforestation, which are harmful to the environment. Deforestation for access purposes, timber and cattle has led to around 3.4 million hectares of woodland being destroyed between 2000 and 2010 and, in turn, soil degradation [1] . As Africa’s rainforest are necessary for global ecological systems, this is a significant loss. Mining and transportation also create damage through pollution and the scarring of the landscape. Mining produces various harmful chemicals which contaminate water and soil, a process which is worsened by illicit groups who cut corners to ensure higher profits [2] .  [1] Food and Agriculture Organization of the United States ‘World deforestation decreases, but remains in many countries’   [2] Kolver,L. ‘Illegal mining threat to lawful operations, safety and the environment’ Mining Weekly 16 August 2013
------------------------------------------------------------
  CORPUS 内容:
    title: ss economic policy international africa house believes africans are worse
    text:  Other countries are hypocritical in expecting Africa to develop in a sustainable way. Both the West and China substantially damaged their environments whilst developing. During Britain’s industrial revolution pollution led to poor air quality, resulting in the deaths of 700 people in one week of 1873 [1] . That said, sustainable resource management has become prominent in some African countries. Most countries in the South African Development Community (SADC) have laws which regulate the impact that mining has on the environment, ensuring accountability for extractive processes. In South Africa, there must be an assessment of possible environmental impacts before mining begins, then the company involved must announce how it plans to mitigate environmental damage [2] . In Namibia, there are conservation zones and communal forests where deforestation is restricted in order to prevent negative environmental consequences [3] .  [1] Environmental History Resources ‘The Industrial Age’ date accessed 17/12/13   [2] Southern Africa Research Watch ‘Land, biodiversity and extractive industries in Southern Africa’ 17 September 2013   [3] Hashange,H.’Namibia: Managing Natural Resources for Sustainable Development’ Namibia Economist 5 July 2013
================================================================================

【第 5 条】
  query-id:  test-digital-freedoms-efsappgdfp-pro04a
  corpus-id: test-digital-freedoms-efsappgdfp-pro04b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    No one will ever actually look at the information  If the concern is privacy then there really should be little concern at all because there is safety in numbers. The NSA and other intelligence services don’t have the time or motivation to be tracking down all of our foibles. [1] If the intelligence agencies are watching everyone then they clearly do not have the personnel to be watching the actual communications. Instead certain things or patterns will raise alarm bells and a tiny number will be investigated more closely.  [1] Walt, Stephen M., ‘The real threat behind the NSA surveillance programs’, Foreign Policy, 10 June 2013,
------------------------------------------------------------
  CORPUS 内容:
    title: e free speech and privacy politics government digital freedoms privacy
    text:  Clearly if no one ever actually looked at any information provided by surveillance then there would be no point in conducting it. Even if it were true that no one looks at any of the data being watched is still an intrusion that affects behaviour. It will affect decisions that are perfectly lawful because there will always be the slight worry that someone who you don’t want to have that information because they will think differently of you will obtain it. When the information is out of your hands you can no longer be certain who will obtain it. [1] Since people have been arrested for the information that has been conducted, clearly sometimes the information is checked and used.  [1] Moore, Mica, and Stein, Bennett, ‘The Chilling Effects of License-Plate Location Tracking’, American Civil Liberties Union, 23 July 2013,
================================================================================

'''


'''
【第 5 条】
  query-id:  test-health-ahiahbgbsp-pro03a
  corpus-id: test-health-ahiahbgbsp-pro03b
  score:     1.0
------------------------------------------------------------
  QUERY 内容:
    Easy to introduce  A ban on smoking in public places would be simple to enforce – it is an obvious activity, and does not require any form of complex equipment or other special techniques . It would largely be enforced by other users of public places and those working there. If it changes attitudes enough, it could be largely self-enforcing – by changing attitudes and creating peer pressure 1 .  1 See Hartocollis, Anemona, “Why Citizens (gasp) are the smoking police), New York Times, 16 September 2010,
------------------------------------------------------------
  CORPUS 内容:
    title: addiction healthcare international africa house believes ghanas ban smoking public
    text:  It would require a large amount of resources for law enforcement to go in to such public places occasionally to see that the ban is being enforced.  It would be easier to enforce conditions relating to the packaging and production of tobacco, which occurs on fewer sites, than ban an activity in certain places which is not so enforceable.
'''