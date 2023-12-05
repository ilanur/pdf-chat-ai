import { ChatOpenAI } from "langchain/chat_models/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { getVectorStore } from "./vector-store";
import { getPineconeClient } from "./pinecone-client";
import { formatChatHistory } from "./utils";

const EUI_THEMES = `Africa||European Union (EU)||Migration and citizenship||Energy and climate||Global governance||Governance and regulation||International relations||Ukraine||Democracy||Climate change governance||Public policy||Political economy||Gender equality||Digital transformation||International human rights law||Human rights||Covid-19||Inequality||European law and institutions||Europe in the world||Political behaviour (parties, elections, voting)||Transnational governance||European integration||Western Balkans||Global, imperial and colonial history||Communication and media||Politics and society||Cultural history||Sustainability||Conflict and peace||Education||Artificial intelligence (AI)||Environment||Energy regulation||Technological change and society||Social justice||Middle East and North Africa (MENA)||Digital media||Digital markets||Banking and finance||Policy analysis||20th century (Twentieth century)||Global history||EU governance||Early modern history||Competition policy||Public international law||Public health, health policy||Protection of fundamental rights in Europe||Law and technology||Intellectual history||Gender and sexuality||EU politics (public opinion, European elections, European Parliament)||International economics||Global economics: trade, investment and development||Colonialism||Political parties||Media pluralism and press freedom||Internet, online platforms||19th century (Nineteenth century)||Welfare and social protection||Transnational history||Economic history||Central and Eastern Europe||Renewable energy||Diversity||Court of Justice of the European Union||Comparative and European constitutional law||Information and disinformation||Competition law||Intellectual property (IP), intellectual property rights||Empires||Sustainable banking and finance||Solidarity, social cohesion||Regulation and standards||Latin America||International law and international relations||History of science||Environmental history||Empirical legal methods, empirical legal studies||Civil society||Public administration||Identity||Modern history||Families and children||Employment||Decolonisation||Welfare state, welfare state reform||Political institutions (electoral systems, constitutional design, legislative politics)||Macroeconomics, monetary policy||Law and economics||Labour migration||International security||International economic law||Globalisation||EU constitutional and administrative law||Displacement, refugees, IDP||Discrimination and marginalisation||Sub-Saharan Africa||Social norms and behaviour||Quantitative macroeconomics`;


const EUI_PAPER =`Autonomous algorithmic collusion:
Economic research and policy implications
Stephanie Assad, Emilio Calvano, Giacomo Calzolari, Robert Clark, Vincenzo Denicolò, Daniel
Ershov, Justin Johnson, Sergio Pastorello, Andrew Rhodes, Lei Xu, Matthijs Wildenbeest
March 16, 2021
Authors are listed alphabetically, as customary in economics:
Stephanie Assad, Queen’s University, Kingston / ON / Canada, assads@econ.queensu.ca
Emilio Calvano, emilio.calvano@unibo.it, 0039 051 20 9 8128, Dipartimento di Scienze Economiche,
University of Bologna, Piazza Scaravilli 2, 40126, Bologna, Italy, CEPR, Toulouse School of Economics
Giacomo Calzolari, giacomo.calzolari@eui.eu, 0039 055 4685 952, 0039 055 4685 954, Department of
Economics, Villa La Fonte, Via delle Fontanelle 18, 50014 San Domenico di Fiesole, Italy, CEPR and
University of Bologna.
Robert Clark, Queen’s University, Kingston / ON / Canada, clarkr@econ.queensu.ca
Vincenzo, Denicolò, vincenzo.denicolo@unibo.it, 0039 051 20 9 8484, Dipartimento di Scienze
Economiche, University of Bologna, Piazza Scaravilli 2, 40126, Bologna, Italyand CEPR.
Daniel Ershov, Toulouse School of Economics, Université Toulouse 1 Capitole / France, daniel.ershov@tsefr.eu
Justin Johnson, justin.johnson@cornell.edu, Samuel Curtis Johnson Graduate School of Management,
Cornell University.
Sergio Pastorello, sergio.pastorello@unibo.it, 0039 051 20 9 8144 Dipartimento di Scienze Economiche,
University of Bologna, Piazza Scaravilli 2, 40126, Bologna, Italy
Andrew Rhodes, University of Toulouse 1 - Toulouse School of Economics (TSE), France
Lei Xu, Bank of Canada, Ottawa / ON / Canada, lei.xu2@gmail.com
Matthijs R. Wildenbeest, Indiana University - Kelley School of Business - Department of Business
Economics & Public Policy, US
2
Abstract.
Markets are being populated with new generations of pricing algorithms, powered with Artificial
Intelligence, that have the ability to autonomously learn to operate. This ability can be both a
source of efficiency and cause of concern for the risk that algorithms autonomously and tacitly
learn to collude. In this paper we explore recent developments in the economic literature and
discuss implications for policy.
Keywords: Algorithmic Pricing, Antitrust, Competition Policy, Artificial Intelligence, Collusion,
Platforms.
JEL Classification: D42, D82, L42
3
1. Introduction
In the last fifteen years the drastic reduction of the cost of computation and data storage has (re-)
activated general interest and significant developments in Artificial Intelligence (AI) and its
market applications. In this paper we investigate the use and consequences of algorithms for
pricing decisions that rely on Artificial Intelligence, “AI-powered algorithms”, using both
experimental tools involving such algorithms and empirical techniques. We argue that, in line
with earlier suggestions in the law literature but in contrast to what many economists have
previously argued, the growing use of such algorithms may increase the likelihood of collusion
in some markets. But we also suggest new methods that may help fight such collusion.
Algorithms are not a new phenomenon in markets. At least since the 1980s, industries like
airlines, hotels and financial markets have relied on these tools for pricing and trading decisions.
Pricing algorithms for “revenue or yield management” can be thought as (possibly very long)
lists of prespecified instructions to act in specific ways for specific contingencies, that the
algorithm then executed (such as for example with Expert Systems). The novelty nowadays is a
new generation of AI-powered algorithms. Their “intelligence” lies in the ability to
autonomously learn how to reach a pre-specified objective in unknown environments without
human intervention. Firms who want to deploy a pricing algorithm do not need to input
information about demand or the strategic context in which this algorithm operates. Given a set
of potential actions (today’s price) for each possible observation (say, previous quantities and
prices), the algorithm is capable of autonomously discovering the profit-maximizing mapping
between what they observe and the price they choose.
The application of autonomously learning algorithms to price goods and services brings about
important policy challenges that are becoming more relevant as pricing algorithms spread in
online and traditional brick and mortar marketplaces. Amazon stresses the possibility and the
benefits of pricing automation in its marketplace with a Selling Partners API service,1 and Chen
et al. (2016) document that more than one third of the best-selling items on Amazon.com were
priced by pricing bots in 2014/2015. The European Commission’s 2017 “Final report on the Ecommerce Sector Inquiry” concludes that “A majority of retailers track the online prices of
competitors. Two thirds of them use software programs that autonomously adjust their own
prices based on the observed prices of competitors.” Offline usage of pricing algorithms is
spreading as well, for example, among gasoline retailers in northern Europe.2 There is a growing
new industry of software intermediaries offering automated pricing services, from turn-key
options that even small sellers can afford to fully customized pricing software for large
companies.3 Many of these repricing companies, such as Kalibrate.com, a2i.com, and Kantify,
explicitly rely on AI as a key characteristic of their algorithms.
1 See https://web.archive.org/web/20201101114000/https://developer.amazonservices.com/
2 See also Sam Schechner, “Why Do Gas Station Prices Constantly Change? Blame the Algorithms,” The Wall Street
Journal, May 8, 2017.
3 See for example, https://web.archive.org/web/20180819175854/https://www.techemergence.com/ai-forpricing-comparing-5-current-applications/.
4
The widespread adoption of algorithmic pricing reflects obvious benefits. Algorithms guarantee
faster and potentially “better” decisions while saving costs. They are more responsive to changes
in supply and demand conditions, which implies better inventory management and reduced
waste, especially for perishable goods. They can also exploit consumer information, providing
potentially highly personalized offers that could increase allocative efficiency. There is a general
consensus that algorithmic pricing has the potential to generate significant efficiency gains and
reduce transaction costs.
However, given the key allocative role that prices play in markets, algorithmic pricing can
generate unintended consequences. Autonomous learning algorithms may learn to price
discriminate on the basis of race or gender or fail to learn effective competitive strategies,
resulting in higher market prices. Algorithms may also end up learning that the best way to
guarantee maximal profits is to decrease competition by, for instance, coordinating with rival
algorithms. Algorithms can make collusive outcomes easier to sustain due to increased ease of
monitoring and quicker detection and punishment of deviations (Ezrachi and Stucke 2015;
Mehra 2016). This is especially a concern in markets with high price transparency and near
perfect monitoring like gasoline retail or the Amazon Marketplace. Algorithmic pricing can also
affect competition if a single intermediary software provider sells their product to multiple
competitors. Such adoption could lead to hub-and-spoke (where the provider acts as the hub of
the sellers, Ezrachi and Stucke 2015) or parallel-use scenarios, with competitors coordinating to
higher prices by delegating choices or relaying information to the same third party. These
concerns are warranted by the statements and observed behaviour of software providers. Some
providers promote their products by suggesting that they optimize for long-term revenues and
avoid price wars (see for example Kantify). In Germany, advertisements show that at least one
company offers their software to multiple stations and brands in the retail gas market.
In this paper we focus on algorithmic collusion. The fact that pricing algorithms may learn to
collude autonomously, without being instructed to do so, and possibly without communication,
opens up new challenging scenarios for market players, platforms and antitrust authorities. .
Antitrust law and enforcement identify violations when colluding parties communicate
explicitly. Currently, algorithms learning to tacitly collude (algorithmic collusion) is not a
violation of antitrust or competition laws. It is crucial to study whether algorithms can learn to
tacitly collude, whether algorithmic collusion does arise in practice, and potential policy
responses to it.
The possibility of algorithmic collusion has not gone unnoticed by competition authorities. The
OECD, the EU Competition Commissioner Vestager, the FTC in the US, the Competition
Market Authority (CMA) in the UK, and the French, German and Canadian competition
authorities all raised concerns about this risk and the need for additional information and
monitoring.4 More recently, authorities have also started to envision policy interventions to
4 For example, see pp.109-111 of the 2019 CMA Furman Report, "Unlocking Digital Competition, Report of the
Digital Competition Expert Panel." Also CMA Research and Analysis Jan. 2021 states “collusion appears an
increasingly significant risk if the use of more complex pricing algorithms becomes widespread.” 
5
address algorithmic collusion. The FTC issued a guidance paper on the use of AI in markets with
indications of desirable properties that AI tools should have to avoid unintended consequences.5
The so called New Competition Tool currently being discussed in the European Union to cope
with digital markets should be designed to account for “oligopolistic market structures with an
increased risk for tacit collusion, including markets featuring increased transparency due to
algorithm-based technological solutions (which are becoming increasingly prevalent across
sectors).”6
The interest in algorithmic collusion by market authorities was anticipated by academic research.
Early accounts of the possibility of algorithmic collusion were discussed by legal scholars, in
particular Ezrachi and Stucke (2016) and Mehra (2016). However, it is only recently that
economists have started to work on this topic. Common wisdom among economists was initially
that algorithmic collusion is not possible or unlikely to arise in practice without explicit
communication.7 Theory models of adaptive learning suggest that tacit collusion is not possible
(Milgrom and Roberts 1990). Some economists suggested that even if tacit algorithmic collusion
is theoretically possible, it is unlikely to arise under dynamic real world conditions (Schwalbe
2018).
We present recent experimental evidence that autonomous collusion between algorithms can
arise in synthetic environments (Calvano et al. 2020). We look at a new generation of
reinforcement learning algorithms (Q-learning) that experiment with random actions as part of
their learning. The dynamic systems induced by such algorithms are very hard to fully
characterize using abstract modelling, except for very simple environments that are not realistic
descriptions of markets. 8 Experimentally, however, it is possible to set up a testing environment
to study how algorithms evolve and interact over time. Experiments allow for perfect
identification in controlled albeit synthetic environments. Our setting features (i) algorithms that
are representative of those likely to be used in practice, and (ii) a realistic simulation of actual
marketplaces, i.e. a virtual market populated with consumers and pricing-algorithms. Using this
approach we observe both market outcomes and their determinants. We find that reinforcement
learning algorithms generate supra-competitive prices and that these higher prices are the result
of tacit autonomous algorithmic collusion: without explicit communication, algorithms learn to
engage in retaliatory pricing.
https://www.gov.uk/government/publications/algorithms-how-they-can-reduce-competition-and-harmconsumers/algorithms-how-they-can-reduce-competition-and-harm-consumers 5 https://www.ftc.gov/news-events/blogs/business-blog/2020/04/using-artificial-intelligence-algorithms
6 Proposal for a Regulation by the Council and the European Parliament introducing a new competition tool,
European Commission, ref. Ares (2020) 2877634.
7
See discussion at the session on Machine Learning, Market Structure and Competition at the 2017 NBER
Conference on AI: https://www.economicsofai.com/nber-conference-toronto-2017.
8 Modelling of learning algorithms with a theoretical approach could offer deep insights on what to expect out of a
repeated interaction among autonomous pricing algorithms. Authors in other disciplines (Brafuss et al 2019, for
example) have attempted this approach using stochastic approximation methods. A recent theoretical literature
has addressed the impact on market outcomes of algorithms that are “hard-coded”, thus having no ability to
explore and learn their behavior with market interactions. See for example, Miklos-Thal and Tucker (2019), Brown
and MacKay (2019).
6
Having established that algorithms can learn to collude in synthetic environments, we present the
first real empirical evidence of widespread algorithmic adoption raising margins and prices
(Assad et al. 2020). Empirically, there are substantial challenges to identifying a causal link
between adoption and collusion. Pricing technology is often highly proprietary and adoption of
new algorithms is rarely observed. Adoption choices are also not random and establishing
causality is important. Moreover, even a causal link between adoption and observable markers of
collusion such as higher prices and margins does not necessarily recover the intentions (i.e.,
strategies). Algorithmic adoption can affect competition but also other factors that change market
prices (i.e., better demand discovery). We use comprehensive high frequency pricing data from
German gasoline retailers to identify the adoption dates of algorithmic pricing technology by
individual gas stations. We then use an instrumental variables approach to establish causality.
We recover the effects of adoption on competition by focusing on pre-existing market structure:
looking at the effects of adoption in monopoly vs. non-monopoly markets, and looking at the
effects of market-wide adoption in duopoly markets. We find that algorithmic adoption increases
margins and prices only for non-monopoly stations. In duopoly markets, margins and prices only
increase if both stations adopt. Together, this suggests that algorithmic pricing has an effect on
competition.
After showing that algorithmic collusion is a credible concern, we explore possible policies that
can mitigate it (Johnson, Rhodes and Wildenbeest 2020). Because of aforementioned concerns
with abstract modelling and the lack of empirical variation in policy, this is also done in a fully
controlled experimental environment. The experimental approach allows a researcher to run a
large set of experiments in a fully controlled environment, under many different market
conditions and different algorithmic designs. This potentially allows her to identify factors that
may make it harder for algorithms to collude. These factors may have to do with the design of
the algorithms or with rules governing the marketplace. For example, many marketplaces have
control over which products consumers consider, and could use this power to guide the
behaviour of algorithms towards procompetitive outcomes. We show that changes in platform
design, such as rewarding firms that cut prices with additional exposure to consumers, may help
curb algorithmic collusion. We also show that policies raising consumer surplus can also raise
platform profits. Overall, thoughtful marketplace design decisions may combat anti-competitive
forces even when perpetrated by algorithms.
The paper proceeds as follows. In Section 2, we describe the experimental approach of
algorithmic collusion has been explored in Calvano et al. (2020). In Section 3 we present the
main findings of Assad et al. (2020) on the actual risk of algorithmic collusion. In Section 4, we
discuss the remedies that platforms may put in place as a reaction of seller’s algorithmic
collusion, as investigated in Johnson et al. (2020). Although we think it is too early to provide
concrete policy recipes as more research is needed, the concluding Section 5 distills some
observations from the existing research to develop a robust policy design.
2. Experimental Analysis for Algorithmic Collusion
Calvano et al. (2020) experiment with algorithms within the context of a workhorse model of
competition in the economics literature: repeated oligopolistic competition where several firms 
7
compete over time with differentiated products. Each firm delegates its pricing to algorithms
whose objective is to maximize the firm’s discounted profit over an indeterminate time horizon.
In each period, the algorithm observes and thus reacts to prices effectively charged in previous
periods by all market participants. After making its choice, it observes the resulting profits
realized in that period. The idea of the experimental approach is to study the behavior that these
AI-powered pricing algorithms learn over time by observing them repeatedly interacting in this
virtual market.
In particular, Calvano et al. (2020) perform experiments using a type of AI called reinforcementlearning, specifically Q-learning. Q-learning is an adaptive approach that allows algorithms to
learn about the strategic environment over time based on their own actions. Although firms do
not need to use AI algorithms to set prices, such algorithms are used in many other areas of
application and it may be reasonable to think that firms may adopt these effective techniques in
the future, if they haven’t done so yet.
A small number of “hyperparameters” characterize the design of the algorithms. One parameter
controls the rate of learning, that is the balance between what the algorithm has learned to date
and the new observations. The experimentation rate governs the attitude to explore, that is the
probability that in a given period the algorithm sets a possibly suboptimal price (given available
knowledge) just to check the reaction of the market, that is of consumers and rivals. Few other
parameters are then used to initialize the algorithms in each simulation, to set the discount factor
embedded in the algorithms, the memory of the algorithms (typically one or two past periods)
and, finally, prices are discretized in a finite price grid. The virtual environment is then
completed with the economic parameters, that specify the number of firms active in the market,
each firm’s cost of production, the preferences of consumers (e.g. with logistic or linear demand
functions) and the degree of differentiation, from homogeneous products to less substitutable
ones. Notice that, while in principle these algorithms can react to a wide variety of inputs /
observations, complexity increases rapidly as one expands the size of those potential information
sets. For this reason, the baseline setup of Calvano et al. (2020) are able to condition their current
price only on the previous period prices.Within this setup it is then possible to run many
simulations for each parametric configuration and for different values of the economic and
hyper- parameters. In this environment, algorithms explore and then learn by mutually
interacting in what is called Multi Agent Reinforcement Learning. Giving the algorithms enough
time to learn their strategies (in other terms, to converge according to some specific convergence
criterion), one can investigate several outcomes. First, the actual prices set by the algorithms.
Second, one can “dig” into the algorithms and study the embedded strategies they have learnt.
Experimental Results
Figure 1 illustrates the price distribution observed across 1000 representative simulations, with
two firms and for given and reasonable economic and hyperparameters.9
 The variability reflects
the fact that different simulations lead to quantitatively different, although as we shall see
9 The assessment of the algorithms is performed after experimentation has concluded and the algorithms have
converged to a stable behavior.
8
qualitatively similar, outcomes. It is instructive to compare these observed prices with two
benchmarks, the monopolist price and the competitive price. The latter is the price that
maximizes each firm’s profit, given the rival’s price. For differentiated products each firm prices
above cost, equating marginal costs to marginal revenues. Crucially, in doing so firms ignore the
negative impact that lowering their price has on their rival’s profit. In the limit case of firms
carrying identical products, the competitive price equals the marginal cost of production. The
former is the price that a hypothetical monopoly owner of all firms in the market would set. Or,
equivalently, is the price that a cartel would agree to charge to maximize the members’ joint
profits. The fact that a monopolist internalizes the impact of lowering prices on all firms is the
reason why the monopoly prices exceed competitive one, to the benefit of consumers. The
distribution shows that the algorithms set prices significantly higher than those in the
competitive benchmark and not too far from the monopolist’s prices.
Figure 1. The distribution of prices charged by reinforcement-learning price algorithms
in the virtual market created in Calvano et al. (2020). The price that would maximize
the firms’ joint profit is just above 1.93. The algorithms routinely learned to collude.
9
Clearly, these prices imply higher profits. In the simulations of figure 1, the algorithms manage
to secure about 80% of the additional profits that they could make beyond the competitive
benchmark if they were to behave as a hypothetical monopolist.
10
But how are these high prices obtained? A first possibility is that the pricing algorithms failed to
learn to compete. If this were the case, a better algorithm or also a human player would promptly
realize that by slightly reducing one’s price it would be possible to exploit the rivals’ high prices
and attract most of the consumers in the market. In this case we should not worry about the
results of simulations like those illustrated in Figure 1. High prices would be just a temporary
phenomenon, wiped away by standard competitive pressure. The alternative is that the
algorithms did instead learn to coordinate their prices and support them with some form of
retaliatory pricing.
To test this conjecture, Calvano et al. (2020) perform experiments with the trained algorithms,
meant to document how these tend to react to rival’s price cuts. Specifically, at the end of each
session, that is after concluding the learning phase, they override one of the pricing algorithms
by forcing it to set a lower price for just one period and then report the behavior in the periods
that follows this “shock.” Figure 2 illustrates the typical pattern. Upon observing the firm 1's
price cut, firm 2 substantially reduces its price in subsequent periods. Firm 1 follows suit as if it
were expecting firm 2’s reaction. This temporary “price war” exhibiting significantly lower
prices gradually comes to an end, with both firms returning to the high prices they were charging
before the exogenous shock. This property of reward (keeping prices high unless a price cut
occurs), retaliatory pricing (for undercutting) and eventual forgiveness (increasing prices back to
pre-deviation) is the hallmark of collusion. The algorithms have learned that undercutting the
other firm’s prices brings forth a war with low profits which ultimately makes any attempt to
deviate from the spontaneous cartel price unprofitable.11
10 Precisely, letting P, Pc, Pm being respectively the average observed profit, the competitive profit and the
monopolist’s profit, the profit gain is measured as (P-Pc)/(Pm-Pc)*100. These measures would be 0% in case of
competitive behavior and 100% in case of monopolistic behavior. 11 More generally, Calvano et al. (2020) also show that the algorithms learned to play equilibrium strategies. Given
the learnt strategy and embedded in other algorithms, a given algorithm learns to set a price that (almost)
systematically and perfectly best responds.
10
Figure 2. After the two algorithms have learned their way to collusive prices, an
attempt to “cheat” so as to gain market share is simulated by exogenously forcing one
of the two algorithms to cut its price. From the “shock” period onwards, the algorithm
regains control of the pricing. The deviation is punished by the other algorithm, so
firms enter into a price war that lasts for several periods and then gradually ends as
the algorithms return to pricing at a collusive level. [Source: Calvano et al. (2020).
Copyright American Economic Association; reproduced with permission of the
American Economic Review.]
This finding is very robust as we will briefly discuss next, and it is remarkable that algorithms
display such a stubborn ability to autonomously learn such a fairly sophisticated collusive
strategy.. In fact, the observed pattern is very much consistent with what theoretical economic
analysis of collusion among rational agents generally predicts.
Robustness.
Calvano et al. (2020) first consider the economic parameters. When a larger number of
independent algorithms maximizing their own profits interact in the market equilibrium prices
reduce, but they are still significantly higher than the competitive level and supported by
collusive strategies. Algorithmic collusion also persists, albeit to a smaller extent when firms
differ in terms of cost efficiency and or quality of their product, exactly as theory predicts.
Similarly, a smaller discount factor reduces prices and profits, in this case down to the
competitive level when the algorithms become dynamically myopic with a nil discount factor.
Algorithmic collusion persists also with different degrees of product differentiation and also with
perfectly substitutable products. A stochastic demand reduces the ability to coordinate on high
prices, but still algorithmic collusion prevails, as well as with a variable market structure where
some firms unpredictably enter and exit the market as they would to for example responding to
their inventories.
11
Calvano et al. (2021) also show that algorithmic collusion copes with more complex economic
environments with imperfect information and imperfect monitoring. In the former case
algorithms privately know the costs of their firms but not those of the other firms which may
differ. In the latter, instead, algorithms are not able to perfectly observe the prices chosen by the
other algorithms as they only observe an imperfect signal. Surprisingly, algorithmic collusion
also adapts to these much more complicated environments.
Algorithmic collusion is also robust to changes in the hyper parameters. That is changes in the
design of the algorithms. Clearly, too short experimentation inhibits learning with algorithms
failing to converge, as with extreme values of the learning rate. However, it is clear that
algorithmic collusion is not the product of a fortuitous choice of these parameters and prevails
over a very broad range.
Other interesting experiments can then be performed with the flexibility of the simulated virtual
market. For example, one can show what happens when a new algorithm enters a market
populated by algorithms that have already performed their learning and ended up with
algorithmic collusion. The question is whether the new entrant learns to exploit the high prices of
the “experienced algorithms” or rather it learns to adapt to their collusive behavior. Interestingly,
it is possible to show that what happens is rather the second possibility with the market ending
up in a new equilibrium with collusive prices (possibly reduced by the presence of an extra firm).
It is also possible to verify to what extent the learnt collusive strategies are specific to the
episodes that the algorithms face in their learning history. This can be done taking algorithms
that have performed their learning in different virtual markets and putting them together in the
same market. Interestingly, in this case their behavior is clearly perturbed showing price wars for
a certain number of initial periods, but very quickly they learn to restore a new collusive
equilibrium with high prices supported with the expectation of punishments of deviations.
Policy Issues and Implications.
Collusion is by no means a new phenomenon. Antitrust authorities have been investigating and
fighting cartels organized by managers all over the world for more than a century and we only
became aware of cartels that get discovered. With algorithmic collusion, there are at least two
important novelties. The first is that algorithms’ ability to autonomously learn to collude is
possible and seems very robust, as discussed above. The second, and probably even more
important observation, is that algorithms autonomously learn to collude, without any instruction
to do so, and they do it silently, without any form of communication. Since managers’ intention
to collude and explicit communications have been the key elements to proving unlawful
collusion with humans, algorithmic collusion poses a fundamental legal challenge. If authorities
discover algorithmic collusion, currently this would not constitute a violation of competition law.
We think that the current state of matters could and probably should be addressed. Here we
mention the possible difficulties that will be confronted with.
The type of collusive strategies that algorithms easily learn as discussed above, could be in
principle adopted by humans too. In fact, the tacitly agreed reward-punishment scheme discussed
above is the typical model of collusion that is taught in economics textbooks as the canonical 
12
mechanism for sustaining a collusive agreement, that clearly cannot rely on explicit contractual
obligations. A crucial difference relates to the potential for gathering evidence in the two
scenarios. With humans there is no way courts and authorities could unveil a tacit collusive
agreement, as they cannot read into the managers’ minds. This is why the current application of
antitrust law is administered on the basis of hard evidence of communication, such as emails and
phone calls. With algorithmic collusion, it would instead in principle be possible to document
the learnt strategies, performing experiments along the lines of the experiment depicted in figure
2. This is not to say that it is going to be an easy task, as we further discuss in Section 5, but it is
at least a potential promising avenue to cope with algorithmic collusion.
3. Empirical Analysis of Algorithmic Collusion
Despite growing theoretical and experimental evidence that commonly used pricing algorithms
can reach tacitly collusive equilibria, a question remains about how real this risk in practice is.
The answer will influence the extent to which competition authorities oversee the adoption of
these technologies (see for instance the UK Digital Competition Expert Panel 2019 Report pp
109-111). Therefore, empirical work investigating the impact of the adoption of algorithmicpricing software is essential. However, any empirical analysis must overcome three important
challenges. First, adoption decisions are typically not publicly observed. Second, adoption is
endogenous because the decision to adopt is correlated with factors that are unobserved to
researchers. Finally, even if adoption can be causally linked with higher prices or margins, it is
not clear whether these can be attributed to changes in competition intensity rather than to other
factors, such as an improved ability to price discriminate.
Assad et al. (2020) address these challenges and provide the first empirical analysis of the impact
of wide-scale adoption of algorithmic pricing solutions, complementing existing theoretical and
experimental works. They take advantage of high-frequency retail gasoline price data from
Germany, where advertising by a leading algorithmic software provider, Danish company a2i
Systems, suggests that algorithmic pricing software has been widely available for adoption since
2017.
Algorithmic software providers claim that their products can help gasoline station owners
"master market volatility with AI-powered precision pricing, respond rapidly to market events
and competitor changes" (Kalibrate.com) and take advantage of "superhuman expertise"
(a2i.com). Software providers stress the ability of their algorithms to incorporate market
conditions and variables such as own and competitor prices, sales volumes, costs and weather
and traffic events. For a given station, an algorithm trains based on historical data. It uses these
inputs and takes in additional "real-time" information such as current weather and traffic patterns
to set prices that maximize station profits. Transactions resulting from these prices are fed back
into the algorithm as new inputs.
Although all software providers focus on the speed and responsiveness of their pricing
algorithms, the exact specifications of algorithms used in the retail gasoline market are unknown.
For example, while most software providers claim to condition on historical own and competitor 
13
prices, it is not known how long their algorithms’ memories are.12 Even the type of machine
learning used (adaptive vs. reinforcement learning) is mostly obfuscated. References in a recent
paper that broadly describes one such algorithm, Derakhshan et al (2016), suggests that they use
reinforcement learning techniques that experiment with random actions to learn the state space as
in Calvano et al. (2020) and Johnson et al. (2020), but it is not stated explicitly.
Regardless of the type of learning algorithms used in this market, widespread adoption could still
facilitate collusive behaviour. The German gasoline retail market is subject to price disclosure
regulations and near perfect price transparency. In such an environment algorithms can make
deviations from collusive conduct easier to detect and punish and help sustain supra-competitive
prices. Advertisements in trade publications also suggest that multiple stations in a single local
market could adopt identical pricing software, raising concerns of hub-and-spoke collusion,
depending on how individualized the algorithms are for each customer.
Identifying Station-Adoption.
A first challenge is that the decision to adopt algorithmic-pricing software is not directly
observed in the data. To identify adopters Assad et al (2020) test for structural breaks in pricing
behaviours related to the use of sophisticated pricing software. The software is advertised to
"rapidly, continuously, and intelligently react" to market conditions; automatically setting
optimal prices in reaction to changes in demand or competitor behaviour; or, to maximize
margins without affecting the behaviour of consumers or competitors. Therefore, following
adoption stations should make more frequent and smaller price adjustments, and should react
more quickly to changes in competitors' prices.
These measures of pricing behaviour line up with what is described in the economic and legal
literature discussing algorithmic adoption. Ezrachi and Stucke (2015) point out the ability for
algorithmic software to increase the capacity to monitor consumer activities and the speed of
reaction to market fluctuations. Mehra (2016) notes the ability of AI pricing agents to more
accurately detect changes in competitor behaviour and more quickly update prices accordingly.13
Assad et al. (2020) use a Quandt-Likelihood Ratio test (Quandt 1960), a method standard in the
economics literature, to identify possible breaks for each station. To minimize false positives, a
station is classified as an algorithmic-pricing adopter if it experiences a structural break in at
least two measures within a short time period (taken to be eight weeks, but robust to alternative
specifications). A large number of breaks are found in all three measures. For example, many
stations go from changing their prices five times per day to ten times per day. Approximately
30% of stations experience structural breaks in more than one of the measures. The majority of
12 Results from the experimental literature suggest memory is short. State spaces become exponentially
larger and the price optimization problem becomes increasingly more complex and unstable with longer
memory. Calvano et al. (2020) and Johnson et al. (2020) both limit algorithmic memory to one period.
13 Chen et al. (2016) identify algorithmic pricing users in Amazon Marketplace by measuring the correlation of user
pricing with certain target prices, such as the lowest price of a given product in the Marketplace.
14
these breaks occur in mid-2017, just after algorithmic pricing software becomes widely
available, suggesting that the measures capture algorithmic-pricing software adoption.
Identifying Causal Effects of Adoption on Margins and Prices.
Having identified which stations adopt algorithmic pricing software and when, Assad et al.
(2020) compare outcomes (margins and prices) of adopting and non-adopting stations. The
challenge faced at this stage is that adoption decisions and timing are likely correlated with
station/time specific factors unobservable to the researcher. For example, stations that hire better
managers could be more likely to adopt the new software, but also different in other dimensions
than worse-managed stations, making it difficult to isolate the effects of adoption. A simple OLS
regression, even one controlling for a large number of station- and time-specific characteristics,
as well as changing local confounders such as weather and demographics, would yield biased
and inconsistent estimates of the effect of adoption on outcomes.
Assad et al. (2020) address this challenge with an instrumental variable (IV) approach. They find
variables (instruments) that shift station incentives to adopt the software independently of their
idiosyncratic unobservable characteristics. The instruments allow them to recover the "true"
causal effect of adoption on outcomes. The main IV is the adoption decision by a station's brand
(i.e., by brand-HQ).14
As in other cases of corporate technology adoption (e.g., Tucker 2008), technology adoption in
retail gasoline happens at two levels: at the brand-HQ level and at the individual station level.
Brands make big-picture decisions about the technology they would like their stations to use, and
provide stations with employee training, technical support and maintenance and subsidies.
Individual station owners make adoption decisions specific to their stations. This involves
incurring investment costs such as pump and Point of Sale (PoS) terminal upgrades. The costs
can be substantial and are not necessarily fully subsidized by the brand. An example is the 1990s
Exxon Mobil (Esso's parent company) brand-wide roll-out of the Mobil Speedpass, a contactless
electronic payment system. BusinessWeek reports that to adopt the technology individual station
owners had "to install new pumps costing up to $17,000--minus a $1,000 rebate from Mobil for
each pump" (BusinessWeek). Partial investment subsidies by brands help explain staggered or
delayed technology adoption in this market. Brand-level decisions should not be correlated with
individual station-specific unobservables.
Since brand adoption decisions are also unobserved Assad et al. (2020) use a proxy for adoption
to instrument: the fraction of a brand's stations that adopt AI pricing. If only a very small fraction
of a brand's stations adopts AI, it is unlikely that the brand itself decided to adopt it. If a large
fraction adopts, it is likely that the brand itself adopted and facilitated adoption by the stations.
14 As a robustness check, Assad et al (2020) consider an alternative set of instruments: annual measures of local
broadband internet availability and quality. Most algorithmic-pricing software are "cloud" based and require
constant downloading and uploading of information. Without high speed internet, adoption is not particularly
useful. Conditional on local demographic characteristics broadband quality should not depend on station-specific
unobservables, but stations are more likely to adopt once their local area has access to reliable high speed
internet.
15
Using brand-adoption as an IV, Assad et al. (2020) examine the effects of adoption on mean
monthly prices and margins, as well as on the distribution of prices and margins. They show that
mean station-level margins increase by 0.7 Euro cents per litre after adoption. Mean margins for
non-adopting stations are approximately 8 Euro cents, so this is a 9% increase in margins.15
Other moments of the margin distribution also generally increase after adoption. Adoption also
causes a 0.5 Euro cents per litre increase in mean prices. There are over 47 million cars
registered in Germany (EuroStat). Assuming that each car has an average tank size of 40 litres
and fills up once a week, universal adoption of algorithmic pricing software could increase total
consumer expenditures on fuel by nearly 500 million Euros per year.
Identifying Effects of Adoption on Competition.
There are many channels, other than through competition, that adoption of algorithmic-pricing
software can change margins. For instance, an algorithm can better detect underlying fluctuations
in wholesale prices or better predict demand. To isolate the effects of adoption on competition
Assad et al. (2020) focus on the role of market structure, comparing adoption effects in
monopoly (one station) markets and non-monopoly markets. If adoption does not change
competition, effects should be similar for monopolists and non-monopolists. They also perform a
more direct test of theoretical predictions by focusing on duopoly (two station) markets. Assad et
al. (2020) compare market-level average margins in markets where no stations adopted, markets
where one station adopted and markets where both stations adopted. In the first market type,
competition is between human price setters. In the second it is between a human price setter and
an algorithm, while in the last it is between two algorithms. By comparing all three market types
they are able to identify the effect of algorithmic pricing on competition.
Findings in Assad et al. (2020) show that outcomes vary based on market structure. First,
adopting stations with no competitors in their ZIP code see no statistically significant change in
mean margins, while those with competitors experience an increase of 0.8 cents per litre and a
rightward shift in the distribution of their margins. These results suggest that algorithmic pricing
software adoption raises margins only through its effects on competition. Second, estimates in
duopoly (two station) markets reveal that, relative to markets where no stations adopt, markets
where both do experience a mean margin increase of 2.2 cents per litre, or roughly 28%. Markets
where only one of the two stations adopts see no change in mean margins or prices. These results
show that market-wide algorithmic-pricing adoption raises margins and prices, suggesting that
algorithms reduce competition. The magnitudes of margin increases are consistent with previous
estimates of the effects of coordination in the retail gasoline market (Clark and Houde 2013,
2014; Byrne and De Roos 2019).
Finally, Assad et al. (2020) explore the mechanism underlying the relationship between
algorithmic pricing and competition by asking whether algorithms are unable to learn how to
compete effectively, or whether they actively learn how not to compete (i.e., how to tacitly
collude). If it is the former, immediate increases in margins should be visible. If it is the latter,
15 Estimates using alternative broadband availability IVs are qualitatively similar to the main estimates but
quantitatively larger.
16
algorithms should take longer to train and converge to tacitly-collusive strategies (Calvano et al.
2020). Assad et al (2020) find evidence that margins do not start to increase until about a year
after market-wide adoption, suggesting that algorithms in this market learn tacitly-collusive
strategies. These findings are in line with simulation results in Calvano et al. (2020).
Policy Issues and Implications.
The findings in Assad et al. (2020) provide the first systematic evidence of the effects of
algorithmic pricing software adoption on competition. From the perspective of competition and
antitrust authorities, they are troubling. Algorithmic pricing software can learn to coordinate,
suggesting that widespread adoption of such software can facilitate tacit collusion and raise
prices and markups. To the best of our knowledge, this occurs without explicit communication
between competitors, making it legal according to current competition laws in many countries.
While the evidence in Assad et al (2020) is particular to retail gasoline markets in Germany, the
same algorithmic pricing software has been adopted in gasoline retail markets around the world.
At a minimum, their results suggest that competition authorities in Germany and elsewhere
should undertake a census of retail-gasoline pricing software to understand the market structure
of the algorithmic software market and the extent of adoption. Such a census can help separate
whether the main effect of algorithmic pricing software on competition comes from multiple
stations in a market adopting the same or different algorithms. Which algorithm competitors
adopt is not directly observed and the two possibilities have different implications for regulators
and policy-makers.
4. Platform Design for Algorithmic Collusion
Online marketplaces such as those operated by Amazon, eBay, and Walmart allow third-party
merchants to set the prices of goods that they sell on the marketplace. The potential for collusive
merchant behavior exists, and there is concern that the growing prevalence and sophistication of
pricing algorithms may facilitate collusion.
What steps, if any, can online retail marketplaces take to fight collusion by third-party merchants
and improve competitive outcomes? This is the question posed in Johnson et al. (2020). In that
article, which we discuss and summarize here, we seek answers using both economic theory and
algorithmic experiments, and use the resulting insights to identify relevant policy issues.
We now sketch the underlying economic scenario we are trying to capture (full details can be
found in the above-mentioned article). Imagine a consumer arrives at an online marketplace and
types a product descriptor into the search tool. Perhaps she is looking for a certain type of
product but is unsure exactly which brands to consider. The platform can influence how many
products she considers, and which products those are, in many ways. For instance, the platform
controls the ranking of products on the search page and how many are on that page. Additionally,
if she clicks on a product to gain more information about it, the platform chooses which
additional products to present on that page, and so on. This overall process might be very 
17
complicated and so to capture the basic idea we suppose that the platform chooses how many
products the consumer considers.
Specifically, there are a total of n differentiated products in a category (we assume a standard
logit model of differentiated-product demand), of which k<n are shown to consumers. We
consider two policies that determine the identities of these k products. The simpler of the two
policies is called Price Directed Prominence (PDP). Under this policy, in each period the
platform shows k of the products with the lowest prices.16 The second policy, Dynamic PDP, is
more subtle and is described below.
What happens when a platform steers demand using these policies? We seek to answer this
question primarily using experiments on AI algorithms. However, as a preliminary step we use
economic theory to frame some of the challenges and potential tradeoffs faced by a platform.
Theory Predictions for Price Directed Prominence in Competitive Markets.
If sellers are not colluding, then individual firms have a strong incentive to cut prices, because
the n-k firms with the highest price are not shown to any consumers. Indeed, theory predicts all
firms will set prices equal to marginal cost. Consumers benefit from these price decreases but are
harmed by the loss of variety presented to them. Therefore, PDP induces a tradeoff for
consumers.
We show that this tradeoff between lower prices and less variety benefits consumers as long as
consumers are shown enough products, that is, so long as k is not too small (indeed, we find that
consumers benefit even if almost two-thirds of firms are not presented to consumers). This
simple and intuitive result is nonetheless powerful as it shows that steering techniques that limit
consumer choice can nonetheless benefit consumers, at least when the market is competitive.
Theory Predictions for Price Directed Prominence in Cartelized Markets.
Now suppose that the n firms in the industry have formed a cartel. We note that there are many
prices on which these firms could collude. To draw a sharp contrast with competitive markets,
we focus on collusion at the prices that maximize the overall profits of the n cartel members.
In stark contrast to what happens in competitive markets, theory predicts that PDP harms
consumers when sellers collude both before and after the implementation of PDP. The reason is
that, when the market is cartelized, PDP does not lead to dramatic price decreases. Instead, the
cartel finds it optimal to reduce prices slightly, and this is not enough to compensate consumers
for the variety loss.
There is a silver lining: showing fewer firms to consumers makes it somewhat harder to sustain a
cartel, meaning that some (but likely not all) cartels may no longer be sustainable. To understand
why, recall that cartels are sustainable when each firm’s short-run gain of deviating from cartel
pricing is smaller than the long-run cost to that firm of starting a retaliatory price war. When
16 In the article we more generally allow the n-k firms with higher prices to receive some demand rather than
none, but the number of consumers who see these products is smaller than the number who see the k lowestpriced products. 
18
fewer firms are shown to consumers, the gains to deviating from a cartel are larger because there
are fewer alternative options being offered to consumers.
This silver lining aside, we also consider a different steering technique, Dynamic PDP, which is
tailored to attack the foundations of collusion more directly than PDP. The basic idea of
Dynamic PDP is to make it more attractive for a firm to deviate from a cartel. It accomplishes
this by making it more difficult for a cartel to punish those who deviate from cartel pricing.
Specifically, under Dynamic PDP a firm that cuts its price today is rewarded not only today (by
being one of the firms shown to all consumers) but also in future periods. The future benefits
come in the form of a “cushion” or “advantage” offered by the platform that makes it easier for
that firm to be shown to all consumers in the future even if rivals retaliate with their own price
cuts. The net effect is that a firm cutting prices today expects also to be shown to consumers in
the future even if rivals undercut it somewhat. In equilibrium, for a properly sized cushion, this
logic leads all firms to compete for the cushion and the final effect is a breakdown in collusion.
Indeed, theory predicts marginal-cost pricing under Dynamic PDP, even when firms would
otherwise form a cartel and even when that cartel would be robust to the simpler PDP technique.
Results of Algorithmic Experiments.
But how do actual AI algorithms behave? To investigate, in Johnson et al. (2020) we perform
experiments using the same type of reinforcement-learning (Q-learning) algorithms discussed in
Section 2.
Briefly, we run the experiments in the following manner. We specify the same demand system as
in our theoretical analysis and then allow our algorithms to interact repeatedly with each other
until their learning converges. The algorithms condition on prices from the previous period (in
principle they might condition on a longer horizon, but our assumption of a single period keeps
the state space to a manageable size).We look at how prices, consumer surplus, and platform
profits are affected by implementing the policies of PDP and Dynamic PDP. We separately
consider both a low and a high level of product differentiation.
Our experiments reveal that AI algorithms do not always behave as predicted by theory. Overall,
however, we find support for the idea that platform-design policies that limit consumer choice
can benefit consumers.
In more detail, our first results involve circumstances where firms value the future highly, that is,
have high discount factors (theory suggests collusion among economic actors is easiest in this
case). Consistent with this, and in line with our predictions, we find that PDP may cause
algorithms to lower their prices but that consumers may still be harmed overall due to the loss of
variety. However, contrary to our theoretical predictions, we find that when the level of product
differentiation is low, PDP lowers prices enough that consumers do benefit.
Although it is encouraging that PDP sometimes benefits consumers when the future is valued
highly, the bottom line is that in this case PDP exhibits mixed success in our algorithmic
experiments. It seems that the AI algorithms we use are sufficiently flexible in their learning that
they are able to maintain very high prices, even when PDP is in place.
19
However, our second policy-design tool, Dynamic PDP, appears to work very well even when
the future is valued highly. We find that AI algorithms drop their prices substantially and that
consumers benefit across a wide array of parameter values.
We also perform experiments using lower discount factors, corresponding to situations in which
the future is not valued as highly. Here we find that PDP by itself can achieve large consumersurplus gains. This is in line with our theoretical predictions in which consumers typically
benefit from PDP if markets are competitive rather than cartelized.
We also use our algorithmic experiments to investigate whether a platform benefits from
adopting steering policies that lower prices. This is important because if not then we might
instead expect platforms only to adopt harmful policies such as those that tend to display firms
with higher prices. Our experiments reveal that platforms can benefit from the policies we
consider. Specifically, when a platform receives a per-unit fee from merchants (as, for example,
Amazon does when a merchant uses its fulfillment services) then the platform benefits when
total sales are higher and we find that our techniques sometimes increase the total units sold.
This is more likely when the same steering also benefits consumers, which makes sense: if prices
fall enough to benefit consumers despite the variety loss, total demand is typically up.
On the other hand, at least for the parameterizations we consider, the total revenue generated by
merchants goes down when we impose our steering policies. Because many platforms receive a
share of revenue from their merchants as a fee, at first this suggests that a platform may hesitate
to impose such policies. However, by lowering prices across its entire platform, we believe a
platform may generate additional total demand; the market size of those who frequent the
platform should increase. Our calculations suggest that the platform may often benefit from
lower prices when this is true.
Aggregate merchant (supplier) profits decrease under these policies, both due to the fact that
prices are falling but also because some consumers are only shown a subset of the available
products. Although we do not systematically study the distribution of merchant profits, we can
say that sometimes asymmetric outcomes are reached, with one merchant earning more than
another. Thus, symmetric learning outcomes are not always reached.
Policy Issues and Implications.
As originally described in more detail in Johnson et al. (2020), several policy implications
emerge from that research.
First, steering techniques that limit consumer choice can benefit consumers, because of how such
techniques influence the strategic decisions of firms. A platform that commits to a policy that
limits variety can compel firms to lower their prices, thereby making consumers better off
despite diminished variety. To be clear, this means that sometimes a platform does not display a
particular product to a consumer even though the consumer would prefer that product to those
that they are shown, and yet consumers still benefit. At the same time, this does not imply that
limiting choice on its own is certain to benefit consumers—for consumers to benefit it is crucial 
20
that such a policy causes firms to make procompetitive decisions that they otherwise would not
have taken.
Second, even when algorithmic collusion might otherwise emerge, platforms may have the tools
to fight back and destabilize a cartel. However, doing so may require more subtle policies. In our
analysis, that more subtle policy is Dynamic PDP, which is specifically designed to influence the
intertemporal tradeoff firm face when deciding whether to remain in a cartel or instead act
competitively.
Third, when more subtle policies are required, such policies may appear to be non-neutral and
yet have positive effects on competition. For instance, under Dynamic PDP, in some periods
particular firms receive preferential consideration from the platform. Importantly, however,
today’s preferential treatment is “earned” in earlier periods by cutting prices in those periods,
and so effective policies may still be non-neutral when viewed from a longer-term perspective.
Finally, there has been debate about whether platforms (especially large, dominant platforms)
should have a legal duty to promote competition on their marketplaces. But a pertinent question
in this debate is whether and how platforms can reasonably achieve this outcome. One
implication of our article is that platforms may indeed have some of the tools they need to do
this. Moreover, in some cases these tools can be fairly simple and related to steering techniques
already employed by most platforms.
5. Concluding Remarks
Economists’ research conducted so far on algorithmic collusion suggests that concern over the
use of pricing algorithms may be warranted. As detailed in Section 2, at least in simplified
experimental settings algorithms can autonomously learn to tacitly collude, and as detailed in
Section 3 there is at least suggestive empirical evidence from the retail gasoline market that the
adoption of such algorithms indeed raises prices. Moreover, under current laws algorithmic
collusion may not even be illegal. However, as shown in Section 4, research also suggests that it
may be possible to limit the harm associated with such collusion by changing the rules by which
algorithms interact on online marketplaces.
The experimental evidence in Section 4 naturally leads to questions about whether there are
broader policy initiatives that might fight any algorithmic collusion. Any such initiatives must be
resilient to the fact that AI algorithms might yield many consumer benefits, for example by
enhancing allocative or productive efficiency, not merely lead to collusive prices. Heavy handed
policies such as banning pricing algorithms may be welfare reducing (as well as nearly unenforceable).
Given the uncertainty about whether pricing algorithms primarily help or harm consumers, and
the very early current stage of research on such questions, we believe it is prudent for regulators
to move cautiously, but to continue moving and learning about the diverse uses of pricing
algorithms in the (very complex) real world.
A better understanding of the market for pricing software may be extremely valuable to
authorities. For example, in a specific market it might be highly informative to know whether 
21
most or all industry participants use the pricing algorithm of the same company, and what
exactly led them to adopt the same software.
Perhaps most importantly, it would be tremendously valuable to authorities to understand in
more detail how different algorithms function. It can often be difficult (especially for outside
observers) to understand how algorithms make decisions. Small decisions by the designers of the
algorithms, including hyperparameter selection, objective function, and the data on which
algorithms are trained, can all have substantial effects on how the algorithms ultimately behave.
There is therefore much scope for the exact functioning and intent of an algorithm to be
obfuscated.
One intriguing possibility is that regulators could gain access to the underlying algorithms and
training data. Such access might allow regulators to gain insights into the design decisions
behind specific algorithms, and to experiment to see how they behave in various settings. A
challenge for this approach is similar to challenges encountered in our studies of synthetic
environments (Calvano et al. 2020 and Johnson et al. 2020). There is no standard “format” by
which algorithms operate; instead they are often customized within a specific IT setting and for a
particular problem faced by a firm. Outcomes may also vary depending on the specific
environment faced by the algorithm (i.e., the pricing strategy and algorithms of their
competitors).
We believe that any investigations into algorithmic pricing must acknowledge that pricing
algorithms that operate on marketplaces cannot be understood in isolation. Instead, they must be
studied jointly along with the rules that the marketplace imposes on the algorithms; these rules
themselves are implemented by algorithms, further complicating the situation.17 Financial
markets, for example, have their own specific trading rules and, being populated by algorithms
that automate trading decisions, are a natural subject for further investigation.
We believe we are in the very early stages of both academic and applied research on pricing
algorithms and collusion. Future research, perhaps collaborative research with computer
scientists and others, is urgently needed.
17 Also, consumers may activate to counter algorithmic collusion. Active algorithmic consumers have been
investigated in Gal and Elkin-Koren (2017).
22
Bibliography
Assad, S. and Clark, R., Ershov, D. and L. Xu (2020), Algorithmic Pricing and Competition:
Empirical Evidence from the German Retail Gasoline Market (2020). CESifo Working Paper
No. 8521, Available at SSRN: https://ssrn.com/abstract=3682021
Barfuss, Wolfram, Jonathan F. Donges, and Jürgen Kurths. 2019. “Deterministic Limit of
Temporal Difference Reinforcement Learning for Stochastic Games.” Physical Review E 99 (4):
243–305.
Brown, Z. Y., & MacKay, A. (2019) Competition in pricing algorithms. Harvard Business
School Working Paper, No. 20-067, November 2019.
Byrne, D. and N. de Roos (2019), Learning to Coordinate: A Study in Retail Gasoline, American
Economic Review, 109(2), 591-619, 2019
Calvano, E., G. Calzolari, V. Denicolo, J. E. Harrington, S. Pastorello, (2020) Protecting
consumers from collusive prices due to AI. Science, Nov. 27.
Calvano, E., G. Calzolari, V. Denicolo, S. Pastorello (2020) Artificial Intelligence, Algorithmic
Pricing, and Collusion, American Economic Review 110, 3267.
Chen, L., A. Mislove, and C. Wilson (2016). An empirical analysis of algorithmic pricing on
Amazon marketplace. In Proceedings of the 25th international conference on the world wide
web. International World Wide Web Conference Steering Committee.
Clark, R. and J-F. Houde. (2013), Collusion with asymmetric retailers: Evidence from a gasoline
price fixing case, American Economic Journal: Microeconomics Vol 5, Issue 3, 97-123.
Clark, R. and J-F. Houde. (2014), The Effect of Explicit Communication on pricing: Evidence
from the Collapse of a Gasoline Cartel, The Journal of Industrial Economics Vol 62, Issue 2,
191-228.
Derakhshan, A., F. Hammer and Y. Demazeau (2016), PriceCast Fuel: Agent Based Fuel
Pricing. In International Conference on Practical Applications of Agents and Multi-Agent
Systems (pp. 247-250). Springer, Cham.
Ezrachi A. and M. Stucke, (2016), Virtual Competition: The Promise and Perils of the
Algorithm-Driven Economy. (Harvard University Press 2016)
Ezrachi, A. and M. Stucke (2015), Artificial Intelligence and Collusion: When Computers Inhibit
Competition, University of Tennessee, Legal Studies Research Paper Series #267, 2015.
Gal, M.S., & Elkin-Koren, N. (2017) Algorithmic Consumers, Harvard Journal of Law &
Technology, 30: 1–45
Johnson, J. , Rhodes, A. , Wildenbeest, M. , 2020. Platform design when sell- ers use pricing
algorithms. Cornell University Working Paper.
23
Klein, T. (2019). Autonomous algorithmic collusion: Q-learning under sequential pricing.
Amsterdam Law School Research Paper.
Mehra, S.K. (2016). Antitrust and the Robo-Seller: Competition in the Time of Algorithms,
Minnesota Law Review, 100: 1323–75.
Miklos-Thal, J., ´ and C. Tucker (2019): “Collusion by algorithm: Does better demand prediction
facilitate coordination between sellers?,” Management Science, 65(4), 1552– 1561.
Quandt, R (1960), Tests of the Hypothesis That a Linear Regression System Obeys Two
Separate Regimes, Journal of the American Statistical Association Vol 55, Issue 290.
Tucker, C. (2008). Identifying formal and informal influence in technology adoption with
network externalities. Management Science, 54(12), pp. 2024-2038.
UK Digital Competition Expert Panel (2019), Unlocking Digital Competition. `;

const CONDENSE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `You are an AI assistant for the EUI (European University Institute) located in Florence, Italy. Use the pieces of context retrieved from the files to answer the user question. You can also give info on files.
If you don't know the answer, just say you don't know or ask for more info. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the EUI.


FILES:
- Staff rules for Admininistrative Staff.pdf - ENG
- Staff rules for Academic Staff.pdf - ENG
- EUI Holidays.pdf - ENG

CONTEXT:
{context}

User question: {question}
Helpful answer:`;


/*
const QA_TEMPLATE = `Assign to the paper the relevent themes from the list provided below, by relevance, or propose new if requested.

THEMES:
${EUI_THEMES}

PAPER:
${EUI_PAPER}

User question: {question}
Relevant themes:`;
*/

/*
const QA_TEMPLATE = `Use the pieces of context retrieved from the papers to answer the user question.
If you don't know the answer, just say you don't know or ask for more info. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the EUI.

CONTEXT:
{context}

User question: {question}
Helpful answer:`;
*/

function makeChain(
  vectorstore: PineconeStore,
  writer: WritableStreamDefaultWriter
) {
  // Create encoding to convert token (string) to Uint8Array
  const encoder = new TextEncoder();

  // Create a TransformStream for writing the response as the tokens as generated
  // const writer = transformStream.writable.getWriter();

  const streamingModel = new ChatOpenAI({
    modelName: "gpt-4-1106-preview",
    streaming: true,
    temperature: 0,
    verbose: true,
    callbacks: [
      {
        async handleLLMNewToken(token) {
          await writer.ready;
          await writer.write(encoder.encode(`${token}`));
        },
        async handleLLMEnd() {
          console.log("LLM end called");
        },
      },
    ],
  });
  const nonStreamingModel = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    verbose: true,
    temperature: 0,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    streamingModel,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_TEMPLATE,
      questionGeneratorTemplate: CONDENSE_TEMPLATE,
      returnSourceDocuments: true, //default 4
      questionGeneratorChainOptions: {
        llm: nonStreamingModel,
      },
    }
  );
  return chain;
}

type callChainArgs = {
  question: string;
  chatHistory: [string, string][];
  transformStream: TransformStream;
};

export async function callChain({
  question,
  chatHistory,
  transformStream,
}: callChainArgs) {
  try {
    // Open AI recommendation
    const sanitizedQuestion = question.trim().replaceAll("\n", " ");
    const pineconeClient = await getPineconeClient();
    const vectorStore = await getVectorStore(pineconeClient);

    // Create encoding to convert token (string) to Uint8Array
    const encoder = new TextEncoder();
    const writer = transformStream.writable.getWriter();
    const chain = makeChain(vectorStore, writer);
    const formattedChatHistory = formatChatHistory(chatHistory);

    // Question using chat-history
    // Reference https://js.langchain.com/docs/modules/chains/popular/chat_vector_db#externally-managed-memory
    chain
      .call({
        question: sanitizedQuestion,
        chat_history: formattedChatHistory,
      })
      .then(async (res) => {
        const sourceDocuments = res?.sourceDocuments;
        //const firstTwoDocuments = sourceDocuments.slice(0, 2);
        //const pageContents = firstTwoDocuments.map(
          const vectorDocuments = sourceDocuments.slice(0, 3);
          const pageContents = vectorDocuments.map(
          ({ pageContent }: { pageContent: string }) => pageContent
        );
        const stringifiedPageContents = JSON.stringify(pageContents);
        await writer.ready;
        await writer.write(encoder.encode("tokens-ended"));
        // Sending it in the next event-loop
        setTimeout(async () => {
          await writer.ready;
          await writer.write(encoder.encode(`${stringifiedPageContents}`));
          await writer.close();
        }, 100);
      });

    // Return the readable stream
    return transformStream?.readable;
  } catch (e) {
    console.error(e);
    throw new Error("Call chain method failed to execute successfully!!");
  }
}
