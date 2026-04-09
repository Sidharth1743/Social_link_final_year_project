

LINKSOCIAL: Linking User Profiles Across
## Multiple Social Media Platforms
## Vishal Sharma
Department of Computer Science
Utah State University, Logan, Utah, USA
vishal.sharma@usu.edu
## Curtis Dyreson
Department of Computer Science
Utah State University, Logan, Utah, USA
curtis.dyreson@usu.edu
Abstract—Social media connects individuals to on-line com-
munities through a variety of platforms, which are partially
funded by commercial marketing and product advertisements.
A recent study reported that 92% of businesses rated social
media marketing as very important. Accurately linking the
identity of users across various social media platforms has several
applications viz. marketing strategy, friend suggestions, multi
platform user behavior, information verification etc. We propose
## L
INKSOCIAL, a large-scale, scalable, and efficient system to link
social media profiles. Unlike most previous research that focuses
mostly on pair-wise linking (e.g.,Facebook profiles paired to
Twitter profiles), we focus on linking across multiple social media
platforms. L
INKSOCIALhas three steps:(1)extract features from
user profiles and build a cost function,(2)use Stochastic Gradient
Descent to calculate feature weights, and(3)perform pair-wise
and multi-platform linking of user profiles. To reduce the cost of
computation, L
INKSOCIALuses clustering to perform candidate-
pair selection. Our experiments show that L
INKSOCIALpredicts
with 92% accuracy on pair-wise and 74% on multi-platform
linking of three well-known social media platforms. Data used in
our approach will be available at http://vishalshar.github.io/data/.
Index Terms—Social Media Analysis, User Profile Linkage,
## Social Media Profile Linkage, Entity Resolution
## I. INTRODUCTION
Social media is an amalgam of different platforms covering
various aspects of an individual’s on-line life, such as personal,
social, professional, and ideological aspects. For instance, an
individual may share professional content on LinkedIn, social
pictures on Instagram, and ideas and opinions on Twitter [16].
A recent study found that more than 42% of the adults use
more than two social media platforms in everyday life
## 1
## .
An individual creates aprofileto participate in a social
media platform. A profile has apublicview and aprivateview,
e.g.,a credit card number would be part of the private view. In
this paper we are only concerned with a publically available
information of a profile that consists of ausername,name,bio
andprofile image. This limited profile is at the intersection of
the kinds of information in public profiles across social media
platforms. An individual has a separate profile for each social
media platform.
Social media platforms generate massive amounts of data.
Previous studies have analyzed this data to learn a user’s
behavior [17], interests [18] and recommendations [19]. But
## 1
http://bit.ly/2FiRy8i
such studies were limited to using only one aspect of an
individual’s on-line life by harvesting data from a single social
media platform. By linking profiles from several platforms
it would be possible to construct a much richer body of
knowledge about a person and glean better insights about their
behavior, social network, and interests, which in turn can help
social media providers improve product recommendations,
friend suggestions and other services.
User Profile Linkage (UPL) is the process of linking user
profiles across social media platforms. Previous research has
shown how to use features in a profile to achieve UPL. For
instance, 59% of the users prefer to keep theirusername
the same across multiple social media platforms [13], which
makes theusernamean important feature in UPL. But ex-
ploiting such features is not straightforward as there can be
inconsistent, missing, or false information between profiles.
UPL is also computationally expensive, making it difficult to
obtain high accuracy in the linkage across platform [32].
We propose a scalable, efficient, accurate framework called
## L
INKSOCIAL, for linking user profiles on multiple platforms.
The framework is depicted in Figure 1. To the left of the
figure, L
INKSOCIALcollects profiles from Google+, Twitter,
and Instagram. The data is cleaned and features of the data
are extracted. Next, the similarity is measured in various
ways, depending on the feature. Preliminary matches are then
refined, and a final match prediction is made. Our empirical
evaluation shows the efficacy of L
## INKSOCIAL.
This paper makes three major contributions.
1) We describe how to engineer relevant features for linking
user profiles across multiple social media platforms. We show
that highly accurate linkage can be achieved by using relatively
few public features.
2) We show how to decrease the high computation cost of UPL
by using clustering. Our intuition is that if we can reduce the
number of linkage attempts, then the cost will decrease, so
we focus on pruning low similarity linkages. Given a user’s
profile from one social media platform, to find similar profiles
from other platforms, we cluster candidate profiles using
similarity based on bi-grams of theusernameandname. Our
experiments show that optimization preserve accuracy while
reducing computation by 90%. The cost reduction claimed
does not include pre-computing cost of Jaccard similarity.
3) We empirically evaluate the effectiveness of our framework
## 260
2018 IEEE International Conference on Big Knowledge
## 978-1-5386-9125-0/18/$31.00 ©2018 IEEE
## DOI 10.1109/ICBK.2018.00042
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.

## Data
## Cleaning
## Feature
## Engineering
## Bio
## User Name
## Full Name
## Similarity
## Measure
## Learning
## Weights
## Match
## Prediction
Jaro - WinklerTF-IDF
KL-Divergence
## SGD*RF*
Fig. 1. LINKSOCIALFramework
by performing extensive experiments and achieving 92% accu-
racy on pair-wise and 74% accuracy on multi-platform linking.
This paper is organized as follows. Section II motivates
the User Profile Linkage problem. In Section III the data
acquisition is described. The L
INKSOCIALframework is intro-
duced in Section IV. The section also describes the engineered
features, how we measured similarity, and how we optimized
for feature weights and reduced the cost of the computation.
Section V reports the results of our empirical evaluation
of L
INKSOCIAL. Related work is presented in Section VI.
Section VII presents conclusions and future work.
## II. M
## OTIVATION
In this section, we describe applications of LINKSOCIAL
and challenges in UPL.
A. Applications ofL
## INKSOCIAL
Data about an individual’s social, professional, personal and
ideological aspects can be used in various ways.
Security- Social media is widely used for spreading malicious
content [21]. Consider a user spreading such content on a
social media platform, their activity can be observed on other
platforms using L
INKSOCIAL. This can help security agencies
identify threats or other malicious activity.
Multi-Platform User Behavior- User behavior and activi-
ties have been studied extensively using single social media
platforms [17]. Linking behaviors from multiple platforms
can create a comprehensive picture of a user’s behavior. For
example, a userAmay be active in social life but disassociate
professionally. Understanding multi-platform user behavior
may lead to insights into why and how friends network
differ across platforms. L
INKSOCIALcan help link different
behaviors to support multi-platform studies.
Information Verification- A user profile could contain
false information. For instance, the mobile social networking
app Skout reported that, on average, every two weeks three
adults masqueraded as teenagers in its forum [22]. By linking
user accounts from multiple social media platforms we can
check consistency and improve verification of information by
merging and validating information from several sources.
Recommendation- Recommendation of products and ser-
vices is usually based on data from a single social media
platform [19]. Data from multiple platforms can enhance the
quality and relevance of recommendations, thus, increases
## 02,000  4,000  6,000  8,000 10,000 12,000
## 0
## 1
## 2
## 3
## 7,432
## 3,348
## 1,112
## 0
## 7,528
## 2,266
## 932
## 232
## 12,163
## 1,798
## 0
## 0
#Count
## Missing Attributes
## Google
## Instagram
## Twitter
Fig. 2. Missing profile information on various social media platforms.
user engagement. Most friend recommendation algorithms
leverage mutual friends. As pointed out by Shu. et al, multi-
platform friend recommendations could improve on pair-wise
recommendations [20].
B. Challenges in linking users cross social media platforms
Data Collection- Gathering profiles from social media plat-
forms is not trivial since user privacy concerns limit the
available information [27]. Even if we could scrape platforms
and collect millions of profiles, ground truthing a UPL solution
is elusive since there is no independent way to verify that
profiles from different platforms belong to the same user.
Incomplete information- The attributes in a public profile
differs across platforms. Some platforms may have an e-mail
address, provide a location, or contact information, but most
do not. Users also incompletely fill out a profile. Some users
may expose gender, age, or ethnicity, but not all users due
to privacy. The inconsistency of profile attributes between
platforms and among users decreases the potential for linkage.
False information- Faking identity on social media is com-
mon [23] as is sharing false content [25] and providing false
information about an individual [24]. Social media platforms
do not provide verification mechanisms for profile data.
Missing information- A profile provides only a small amount
of data about a user since privacy concerns limit the amount
of public sharing [26]. Profiles with missing data further
exacerbates the difficulty of linking as shown in Figure 2. The
paucity of available data coupled with the high rate of missing
data complicates the task of accurately linking profiles.
Limited Access- Data from major social media platforms
can be accessed through a platform-specific API but due to
privacy concerns, social media providers reveal only a limited
## 261
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.

amount of data. Even after collecting data, we might lack
enough common features to link profiles.
## III. P
## ROBLEMDEFINITION
This section gives the problem statement, and discusses data
collection and pre-processing.
## A. Problem Statement.
LetP
k
i
be thei
th
public profile on social media platform
k. LetIbe an identify function that maps aP
k
i
to the identity
of the user who created the profile. For linking profiles from
nsocial media platforms, we use the following objective
function.
## Φ
n
## (P
## 1
i
## ,...,P
n
k
## )=
## {
1ifI(P
## 1
i
## )=...=I(P
n
k
## )
## 0otherwise
## (1)
Our goal is to build and learn functionsΦ
## 2
(.)andΦ
## 3
## (.)for
linking pair-wise and three-platform profiles. We assume that
in our dataset, every user has exactly one profile in each social
media platform.
## B. Dataset Collection
For testing a UPL solution, we need ground truth data.
While social media platforms have APIs to access user data,
there is no platform to help us link a profile to an individual.
However, there are ways that we could build a set of ground
truths. For instance, we could crowd-source the ground-
truthing (viz. Amazon Mechanical Turks) or use surveys [22],
but these methods are prone to getting unreliable data. Instead,
we used a novel resource, the websiteabout.me, which is a
site that requires users to input links to profiles on other social
media platforms . When creating a profile on the site, a user
will provide links to their other social media profiles.
We used a dataset of 15,298usernamesfrom six social me-
dia platforms: Google+, Instagram, Tumblr, Twitter, Youtube,
and Flickr [12]. We narrowed the dataset to Instagram, Twitter,
and Google+ for our study since they make theusername,
name,bio, andprofile imageare publicly available.
Next, we built a crawler to collect profiles from the three
platforms. Table II displays information about our collected
data. We gathered data on 7,729 users from all three platforms,
6,039 users have data available on a pair of platforms and
1,530 have data available only from one platform. Missing
profiles could be because of users deactivating their accounts.
## C. Dataset Analysis
We analyzed the collected profiles to determine how much
information was missing. Figure 2 shows the count of missing
profile attributes for each platform. In Google, Twitter, and
Instagram there were 28%, 13% and 21% of user profiles
with at least one missing attributes, respectively. 9% and 8%
of Google and Instagram profiles had at least two missing
attributes. There were 87%, 62%, and 69% of profiles on
Twitter, Google, and Instagram, respectively, without miss-
ing information. Three attributes were missing from 2% of
Instagram profiles, making it impossible to match them. The
attributes that were presented have some variance. On average,
## TABLE I
## N
## UMBER OF PROFILES OBTAINED PER SOCIAL MEDIA PLATFORM.
Social MediaProfile Count
## Instagram10,958
## Twitter13,961
## Google+11,892
## TABLE II
## N
## UMBER OF USERS WITH PROFILES ON PAIR-WISE AND
## MULTI
## -PLATFORMS
Social MediaProfile Count
Instagram-Google+614
Twitter-Instagram2451
Google+-Twitter2974
Google+-Instagram-Twitter7729
a
There were 1530 users with only one profile.
bioson Google+ were longer than Instagram and Twitter;
164 characters on Google+ compared to 70 and 96 for Insta-
gram and Twitter. However, theusernameattribute has little
variance, it was 11-13 characters on average across all three
platforms.
## IV. T
## HELINKSOCIALFRAMEWORK
This section describes LINKSOCIAL, discusses computation
cost, and shows how to reduce the cost using clustering.
## A. Feature Engineering
## L
INKSOCIALuses the basic features in a profile as well as
the following engineered features.
Username and Name bi-grams- Theusernameis an
important feature for profile linkage since people tend to use
the sameusernameacross social media platforms. When a
new name is used, parts of the name is often re-used. For
example,John_snow,johnsno, andsnow_johncould
belong to the same person. String similarity metrics such as
Jaro-Winkler, longest common subsequence, or Levenshtein
distance, tend to perform poorly onnametranspositions [34].
To better align names, we engineer the bi-grams ofusernames
as a feature. We also merge the bi-grams ofusernameswith
the bi-grams of names as a feature since people also like to
transpose their surname and first name in for ausername.We
engineer the following feature sets.
•bi-gram ofusername.(u
u
## )
•bi-gram ofname.(u
n
## )
•merging above two features. (u
b
## )
These three feature sets capture a range of different ways to
create ausername.
Character Distribution- User’s like to createusernames
using substrings of theirnameor other personal information
(a pet’s name or a significant date). To handle scenarios where
bi-grams could not capture the similarity, we use the character
distribution ofusernamesandnamesas features. To measure
distribution similarity we use Kullback-Leibler divergence as
defined below,
## KL
divergence
## (P||Q)=
n
## ∑
i=1
## P(i)·log
## P(i)
## Q(i)
## (2)
wherePandQare given probability distributions. We engi-
neered the following features sets using character distribution
similarity.
## 262
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.

## •username.(u
usim
## )
## •name.(u
nsim
## )
## •username+name.(u
bsim
## )
We perform experiments on real data and have not consid-
ered scenarios whereusernameandnamehave no relationship.
Profile Picture Similarity- Users often use the same
profile picture in multiple platforms. To capture similarity
betweenprofile images, we use OpenFace [14], an open source
image similarity framework. Openface crops images to extract
face and uses deep learning to represent the face on a 128-
dimensional unit hypersphere. We use
## 2
norm to calculate
distance between vectors of twoprofile images.
## B. Similarity Measures
## L
INKSOCIALuses two similarity measures.
Jaro-Winkler Distance- Jaro-Winkler is metric for string
matching and is commonly used when matchingnamesin
UPL [11]. Studies show that the metric performs well for
matching string data [15]. Jaro accounts for insertions, dele-
tions, and transpositions while Winkler improves Jaro based
on the idea that fewer errors occur early in aname.
TF-IDF and Cosine Similarity-L
INKSOCIALuses a
different similarity measure for matching profilebiossince
biosare longer thannames. TF-IDF and cosine similarity are
widely used for measuring the similarity of documents.
## C. Matching Profiles
## L
INKSOCIALmatches profiles using the basic and en-
gineered features of a profile. We transform the matching
problem into a multivariate binary classification problem and
optimize it using Stochastic Gradient Descent (SGD). It is
defined as follows:
h(x)=w
## 0
## +w
## 1
## ·x
## 1
## +w
## 2
## ·x
## 2
## +w
## 3
## ·x
## 3
## +...+w
m
## ·x
m
## (3)
In the equation,x
## 1
## ,x
## 2
, ..., x
m
represents the similarity score
of features between two profiles andw
## 1
## ,w
## 2
, ..., w
m
represents
their respective weights or coefficient. We use Mean Squared
Error (MSE), as our loss function. Considering the predicted
values ash
w
## (x)
## (i)
wherei∈1,2, ..., nandy
## (i)
asagiven
value (either a match (1) or no match (0)), we defineMSE
or cost functionf
cost
(W)as follows:
MSE=f
cost
## (W)=
## 1
## M
m
## ∑
i=1
## (
y
## (i)
## −h
w
## (x)
## (i)
## )
## 2
## (4)
## L
INKSOCIALuses SGD to optimize the cost function.
Partial derivatives of Equation 4 w.r.t tow
## 1
## &w
## 2
are defined
as follows:
## 
w
## 1
f
## ′
cost
## (w
## 1
## )=
## 1
## M
m
## ∑
i=1
## −2x
## 1
## (
y
## (i)
## −h
w
## (x)
## (i)
## )
## 
w
## 2
f
## ′
cost
## (w
## 2
## )=
## 1
## M
m
## ∑
i=1
## −2x
## 2
## (
y
## (i)
## −h
w
## (x)
## (i)
## )
and similarly for other weights.
Derivatives are followed by updating of the weights. For
example,w
## 1
## &w
## 2
as shown below:
w
## 1
## =w
## 1
## −η
w
## 1
f
## ′
cost
## (w
## 1
## )
w
## 2
## =w
## 2
## −η
w
## 2
f
## ′
cost
## (w
## 2
## )
In the above equation,ηis the learning rate, a value that
typically ranges between 0.0001 - 0.005. During experiments,
we also add elastic net regularization for training.
The derivatives and weights are recursively calculated until
the equation converges and yields an optimized weight for
each attribute based on a training set. To find a match of a
given profile we use Equation 5 where we find the profile
with maximum score on the given attributes and weights.
In Equation 5,W
## T
is a weight vector calculated using the
optimization of Equation 4 andX
u
is a vector of attributes.
## P
k
i
is a profile from social media platformk, whileU
j
is a
set of all profiles from platformj.
Match(P
k
i
## ,U
j
)=max(W
## T
## ·X
u
),∀u∈U
j
## (5)
Given profileP
k
i
,Match()outputs the most similar profile
from platformj.
## D. Computation Reduction Using Clustering
UPL can be computationally expensive. Given our dataset
with 7,729 user profiles, if we are linking pairs of profiles from
only two of the platforms, then the number of comparisons will
be7,729∗7,728 = 59,729,712. Assuming we can perform
1,000 comparisons per second (which is a very high ballpark),
it will require 17 hours to perform UPL. Matching of millions
of users across multiple platforms will be infeasible (without
the dedication of massive computing power).
To tackle this problem, we introducecandidate profile
clustering. We can reduce the number of comparisons by
pruning low potential comparisons, that is, by avoiding the
work of matching profiles that are dissimilar. In our dataset,
45% of the Instagram and Twitter profiles have the same
usernameandname. By clustering on the bi-gram features for
usernameandnamewe can prune comparisons from profiles in
different clusters. We have observed in previous studies [35],
Jaccard similarity relativly performs well for finding similarity
between bi-grams/n-grams and is also computationally not
expensive. We rank profiles based on Jaccard similarity and
we choose the top 10% of the candidate profiles with the
highest score in the cluster. Algorithm 1 gives our approach
for building clusters. Our clustering approach is conceptually
similar to kNN clustering where distance is defined by Jaccard
similarityj
simandkis 1.
Clustering of profiles before linking can help reduce compu-
tation cost significantly buthow big should our clusters be?If
the cluster is big, the computation cost will increase; if cluster
is small, we may fail to capture maximum profile matches. To
understand the effect of clustering, we plotted cluster size (as
a percentage of the total number of profiles) versus match
accuracy on our training data as shown in Figure 3. In the
plot, thex-axis is the percentage of profiles in a cluster (on
## 263
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.

## 1  5   10  15  20  25  30  35  40  45  50
## 75
## 80
## 85
## 90
## 95
## 100
## .
## Cluster Size (%)
## Accuracy (%)
## Accuracy
Fig. 3. Comparison of Cluster size to Accuracy on train data.
average). They-axis shows the maximum accuracy we can
achieve since the matching profile has to be in the cluster.
We observed that clusters roughly 10% of the size of the data
reduces 90% of computation cost while preserving∼90% of
the potential matches.
Algorithm 1Computing candidate profiles for a given useru
Input:u,UUis set of all users from a different social media
thanu.
1:procedureFINDCANDIDATEPROFILE(u,U,n)nis number
of profile to find in cluster.
## 2:u
bg
## ←u
u
## +u
n
3:for eachp∈Udopis single user profile
## 4:p
bg
## ←p
u
## +p
n
5:jsim←Jaccardsimilarity(u
bg
## ,p
bg
## )
6:Score[p]←jsimInsert in dictionary
7:end for
8:CandidateProfiles=gettopn(n, Score)
9:returnCandidateProfiles Returnsncandidate profile
similar tou
10:end procedure
11:procedureJACCARDSIMILARITY(u
bg
## ,p
bg
## )
12:Intersection=|u
bg
## ∩p
bg
|Number of common
elements.
13:Union=|u
bg
## ∪p
bg
|Number of unique elements.
14:JaccardSimilarity=
## Intersection
## Union
15:returnJaccardSimilarity
16:end procedure
17:procedureGETTOPN(n, Score)
18:Score←Sort(Score)Sort Score w.r.t value
19:topkey=getkey(Score,n)returns top n key from
sorted Score
20:for eachkey∈topkeydo
21:topn←Score[key]
22:end for
23:returntopnreturn top n candidate profiles
24:end procedure
## V. EXPERIMENTS
This section reports an experimental evaluation of LINKSO-
CIAL. We establish a baseline for UPL and verify that
## L
INKSOCIALcan learn Equation 5 and achieve high matching
accuracy. We use several variations of calculating weights and
compare them with the baseline to determine the impact of
the weight calculations. Specifically, we use Random Forest
(RF) and Stochastic Gradient Descent (SGD) for calculating
weights. We also performed feature analysis to understand
which features are important.
## TABLE III
## C
## OMPARISON WITH PREVIOUS RESEARCH
## Linkage
AuthorsReduceScalable
et. alFeaturesDatasetScalableCostAcross
PublicPlatform
Pair-Wise
P. Jain[28]Private
A. Mal[8]Public
R. Zafa[22]Public
Y. Li[31]Private
LINKSOCIALPublic
## Across
X. Mu[32]Private
S. Liu[6]Private
LINKSOCIALPublic
## A. Experimental Setup
We used the data described in Section III-B. We chose a
60-40 split in the data for training and testing.
Baseline- We built a baseline using Jaro-Winkler and TF-
IDF as discussed in Section IV-B. We use Jaro-Winkler to
analyzeusernameandnamesimilarity, and TF-IDF and cosine
similarity for profile bios. To find the match of a user profile,
we select a profile with the highest score from Equation 6
where the value of theWvector is 1, considering each feature
is equally important andXrepresents the feature vector.
f(W)=W
## T
## ·X
u
## (6)
Calculating Weights- To calculate weights for pair-wise
linking, we generated all the features discussed in Sec-
tion IV-A for each pair, which gives us data for correct
matches. To generate data for mismatches, we randomly chose
pairs equal to the number of correct matches and collected
their feature scores.
We used normalized variable importance score from RF and
also SGD optimization algorithm as explained in Section IV-C
for weights calculation. RF was trained using 10-Fold Cross
Validation, it was tuned using grid search, and the square
root of the number of features was selected as the number
of variables to be randomly sampled as candidates at each
split. SGD was also performed on the same dataset to calculate
weights with a learning rateηof0.001, with 1,000 iterations.
Computing Candidate Profile- To compute candidate
pairs, we follow the approach described in Algorithm 1. In
training, we pre-compute each profile’s potential matches. To
generate a feature vector we score based on clusters. To make
sure we have both positive and negative label samples of equal
size, we randomly sample negative label data of the same size
as the positive label set and generated samples are used to
learn the weight vector.
Multi-Platform Profile Linking- We link users across
three platforms similar to how we perform pair-wise linking.
Due to the very high computation cost, we were unable to
run experiments for linking without first clustering candidate
profiles. We performed experiments by adding and removing
engineered features. To find a similar profile, we choose a
profile in one social media platform and tried to find the
matching profiles in the other two platforms. We performed
## 264
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.

## TABLE IV
## L
## INKSOCIAL PERFORMANCE ON PAIR-WISEUPL
Social Media Pairs (Accuracy %)
ExperimentsG+≡IT≡IG+≡T
baseline55.36%77.86%56.86%
Prediction without engineered features and clustering.
with RF77.53%82.08%77.14%
with SGD76.61%82.21%66.24%
Prediction with clustering, no engineered features.
with CP & RF82.62%83.33%81.40%
with CP & SGD82.47%83.32%81.19%
Prediction with engineered features, no clustering.
with RF86.54%91.17%84.56%
with SGD86.63%91.68%84.58%
Prediction with engineered features and clustering.
with CP & RF84.85%87.92%83.20%
with CP & SGD84.91%88.29%83.23%
this experiment for each platform. During training, we choose
a platform and compute its respective candidate profiles from
other platforms by building feature vector between candidate
profiles and the given profile. We then perform SGD and RF to
calculate weights for each feature and we used the calculated
weights to find similar profiles.
Previous research comparison- Table III compares our
approach to previous approaches in terms of feature selection,
data availability and scalability. Previous approaches have
used different types of features in their algorithms. Apublic
featureis publicly available information (e.g., bio,profile
image,name,user name).Privatedata such aslocation
data is proprietary limiting the feasibility of approaches that
rely on such information since they need cooperation from
businesses that compete against each other. Making a publicly
available benchmark dataset is an important aspect for future
work and comparison of approaches. Previous researchers
have not shared their data (because of privacy concerns and
data sources), where as in our approach we have collected
all data as public information and have made it available
for future research/comparison. As discussed earlier, UPL is
a computationally expensive process and scalability of an
approach depends on using methods to reduce computation
cost to make the solution practical in a real-life scenario. Also,
designing an algorithm to scale for new/several platforms is
very important aspect. Only a few approaches in past have
used computation cost reduction and are designed to scale for
new social media platforms.
Previous work accuracyTable VI reports accuracies in a
sampling of previous research in both pair-wise and multi-
platform linkage. The accuracy is reported “as is” from the
papers, experimental setups and measurements differ across
papers, for instance previous work with better accuracy for
pair linkage used user generated content data (gaining access
to such data is difficult) but in our approach we use only
publicly available profile data We observe that L
## INKSOCIAL
is among the leaders in pair-wise UPL and the best at multi-
platform UPL.
## TABLE V
## L
## INKSOCIAL PERFORMANCE ON MULTI-PLATFORMUPL
## Cross Platform
ExperimentsT→(G+,I)G+→(T,I)I→(G+,T)
## CP&RF71.56%72.50%73.70%
## CP & SGD72.95%72.86%74.18%
*RF−Random Forest, SGD−Stochastic Gradient Descent, CP−Candidate
Profiles using Clustering, T−Twitter, G+−Google+, I−Instagram
## TABLE VI
## R
## EPORTEDACCURACY FROM FEWPREVIOUS WORK.
LinkageAuthorsAccuracy
Pair-WiseP. Jain et al. [28]39.0%
SocialA. Malhotra et. al. [8]64.0%
PlatformR. Zafarani et. al. [22]93.8%
LinkageY. Li et. al. [31]89.5%
## Our Approach91.68%
AcrossX. Mu et. al. [32]44.00%
MultipleS. Liu et. al. [6] (reported by [32])42.00%
PlatformOur Approach74.18%
## B. Evaluation Metrics
In previous studies, accuracy has been used as a reliable
evaluation metric for UPL [28]. Given two profiles from
different platforms, the accuracy of such matching can be
measured as follows. First, assume the following are known.
•Total number of correct prediction(P): Number of correct
positive prediction by L
## INKSOCIAL.
•Total number of positive sample(N): Number of positive
linked profiles in the dataset.
Then the accuracy can be computed as follows.
## Accuracy(%) =
## |P|
## |N|
## ·100(7)
## C. Results
We performed several experiments on pair-wise linking and
the results are shown in Table IV and Table V. Specifically,
we measured the accuracy of L
INKSOCIALon all possible
pairs in our dataset namely, Google+ - Twitter, Google+
- Instagram, and Twitter - Instagram. We also performed
experiments with features and weights calculated using RF
and SGD. As shown in Table [IV], we started with building
a baseline for each pair. We achieved 55%, 78%, 57% for
Google+ - Instagram, Twitter - Instagram and Google+ -
Twitter respectively. We then performed experiments without
using engineered features and clustering. We observed that
RF produced more accurate matches than SGD. Next, we
added candidate pairs but sill no engineered features. In this
case, both RF and SGD performed equally well. We then
added engineered features and perform experiments without
clustering. We observed that SGD outperformed RF. Finally,
we used both engineered features and clustering. SGD again
performed better than RF. Overall, weights calculated using
SGD proved to be more accurate than RF though in some cases
the difference was marginal. In the final stage, we observed
reduction of accuracy by adding clustering. This is due to
the inconsistentusernameandnameused by a user, since
clustering usesusernameandname(in our dataset, out of all
## 265
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.

## 0.3
## 0.4
## 0.5
## 0.000.250.500.751.00
## User Name Bigram Similarity Score
yhat
## 0.2
## 0.3
## 0.4
## 0.5
## 0.000.250.500.751.00
## Merged Bigram Similarity Score
yhat
## 0.48
## 0.50
## 0.52
## 0.000.250.500.75
## Merged Distribution Similarity Score
yhat
## 0.30
## 0.35
## 0.40
## 0.45
## 0.50
## 0.000.250.500.751.00
## Full Name Bigram Similarity Score
yhat
## 0.30
## 0.35
## 0.40
## 0.45
## 0.50
## 0.000.250.500.751.00
## User Name Similarity Score
yhat
## 0.35
## 0.40
## 0.45
## 0.50
## 0.000.250.500.751.00
## Full Name Similarity Score
yhat
## 0.48
## 0.49
## 0.50
## 0.00.20.40.6
## Full Name Distribution Similarity Score
yhat
## 0.4950
## 0.4975
## 0.5000
## 0.5025
## 0.5050
## 0.00.10.20.30.40.5
## User Name Distribution Similarity Score
yhat
Fig. 4. Partial Dependence Plots for individual features.
## 0.0
## 0.1
## 0.2
## 0.3
## 0.4
## 0.5
## 0.00.20.40.60.8
## Merged Distribution Similarity Score
## User Name Distribution Similarity Score
## 0.48
## 0.50
## 0.52
## 0.54
## Partial
dependence
## 0.00
## 0.25
## 0.50
## 0.75
## 1.00
## 0.000.250.500.751.00
## User Name Similarity Score
## Full Name Similarity Score
## 0.3
## 0.4
## 0.5
## Partial
dependence
Fig. 5. Partial Dependence Plots for selected pairs of features.
pairs, maximum of 45% users have the sameusernameand
22.2% of users with the samename), but slight decrease in
accuracy could be traded for speed.
We performed several experiments on multi-platform link-
ing and the results are shown in Table V. We observed
that, feature weights computation using SGD again outper-
formed RF. We achieved an accuracy of 73%, 73%, 74%
for Twitter→(Google+, Instagram), Google+→(Twitter, Insta-
gram), and Instagram→(Twitter, Google+) respectively. Over-
all, in our experiments on multi-platform linkage, SGD proved
to more accurate.
Model Interpretation- To understand our model, we
performed feature analysis using Partial Dependence Plots
(PDP) [29]. Given predictor variables and response variables,
PDP can interpret the relationship between them. We start by
studying the effect of all variables and later choose variable
pairs for further study.
In Figure 4, thex-axis represents feature similarity score
and they-axis represents the effect on the class probability.
In Figure 4 we observed, that the score from the merged
distribution similarity is positively correlated to the model,
usernameandnamesimilarity contributes to the model until
the value of 0.75 then it drops. We plot selected variable pairs,
to study their effect on the model with the results show in Fig-
ure 5. In Figure 5, thex-axis andy-axis represent the score of
respective feature similarity and Partial Dependence represents
the marginal effect of features on the class probability. We ob-
served that higher values of merged andusernamedistribution
similarity score together are highly correlated to the model.
Similarly,usernameandnamesimilarity score values until
0.75 are highly correlated but the highest values are relatively
low. This implies that there are several instances in our data
whereusernameandnamesimilarity scores together are very
high (close to 1), but selected profiles do not belong to the
same individual. We also observed, with their relatively lower
values of similarity, that there are instances where both profiles
belong to the same person. Finally, we concluded that the
similarity ofusernameandnameare insufficient or unreliable
features for linking profiles.
## VI. R
## ELATEDWORK
UPL is a subproblem of a larger problem that has been
studied under different names such as record linkage, entity
resolution, profile linkage, data linkage, and duplicate de-
tection. In the field of databases,entity resolutionlinks an
entity in one table/database to another entity from another
table/database,e.g.,when linking data from the healthcare
to the insurance data [2]. Entity resolution has been referred
to ascoreference resolution[1] in NLP andnamed disam-
biguation[3] in IR. Approaches to solving the problem fall
into three categories: numerical, rule and workflow-based [20].
Numerical approaches use weighted sums of calculated fea-
tures to find similarity. Rule-based approaches match using a
threshold on rule for each feature. Workflow-based approaches
use iterative feature comparisons to match.
There have been several approaches that utilize user be-
havior to solve the pair-wise matching problem, such as, a
model based approach [6], a probabilistic approach [5], a
clustering approach [7], behavioral approach [28], user gener-
ated content [31], and both supervised [9] and unsupervised
learning approaches [10]. The problem of user linkage on
social media was formalized by Zefarani et al. [13] where
they used usernames to identify the corresponding users in
different social community. In our framework, we used a
supervised approach, and mitigate cost by reducing the number
of comparisons.
Most previous work in UPL focuses on pair-wise matching
due to challenges in computational cost and data collection.
In pair-wise matching, Qiang Ma et al. [4] approached the
problem by deriving tokens from features in a profile and used
regression for prediction; R. Zefarani et al. [22] used username
as a feature and engineered several other features by applying a
supervised approach to the problem. Unlike these approaches,
## L
INKSOCIALcan perform multi-platform matching as well
as pair-wise matching. Multi-platform UPL has received less
attention. Xin et. al [32] approached the multi-platform UPL
using latent user space modeling, Silvestri et.al [33] uses
attributes, platform services, and matching strategies to link
users on Github, Twitter, and StackOverFlow; Gianluca et.
al [30] leverages network topology for matching profiles
## 266
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.

acrossnsocial media. Liu et. al. [6] uses heterogeneous user
behavior (user attributes, content, behavior, network topology)
for multi-platform UPL but gaining access to such data is not
a trivial task.
## VII. C
## ONCLUSION
In this paper, we investigate the problem of User Profile
Linkage (UPL) across social media platforms. Multi-platform
linkage can provide a richer, more complete foundation for
understanding a user’s on-line life and can help improve
several research studies currently performed only on single
social media platform. UPL has many potential applications
but is challenging due to the limited, incomplete, and po-
tentially false data on which to link. We proposed a large
scale, efficient and scalable solution to UPL which we call
## L
INKSOCIAL.LINKSOCIALlinks profiles based on a few
core attributes in a public profile:username,name,bioand
profile image. Our framework consists of(1)feature extraction,
(2)computing feature weights, and(3)linking pair-wise and
multi-platform user profiles. We performed extensive experi-
ments on L
INKSOCIALusing data collected from three popular
social media platforms: Google+, Instagram and Twitter. We
observed thatusernameandnamealone are an insufficient
set of features for achieving highly accurate UPL. UPL is
computationally costly, but we showed how to use clustering to
reduce the cost without sacrificing accuracy. Candidate profile
clustering is based on pruning dissimilar profile comparisons.
It reduced 90% of the comparisons which significantly helped
in scaling our framework. We evaluate our framework on both
pair-wise and multi-platform profile linkage with accuracy
91.68% on pair-wise and 74.18% on multi-platform linkage.
Data about a user from multiple social media platforms
has many applications. In future, we plan to(1)extend our
work to study a user’s behavior across platforms, which
to our knowledge has not yet been studied,(2)add more
features to L
INKSOCIALusing heterogeneous data,e.g.,user
content similarity (text, videos, images), network similarity,
and patterns across social media platforms, and(3)evaluate
## L
INKSOCIALon more (up to six) social media platforms.
## R
## EFERENCES
[1] J. Cai, M. Strube, “End-to-End Coreference Resolution via Hypergraph
Partitioning,” COLING, 2010.
[2] S. E. Whang, D. Marmaros, H. Garcia-Molina, “Pay-As-You-Go Entity
Resolution,” IEEE Transactions on Knowledge and Data Engineering,
## 2013.
[3] Y. Qian, Y. Hu, J. Cui, Q. Zheng, Z. Nie, “Combining machine learning
and human judgment in author disambiguation,” CIKM, 2011.
[4] Q. Ma, H. H. Song, S. Muthukrishnan, A. Nucci, “Joining user profiles
across online social networks: From the perspective of an adversary,”
## IEEE/ACM ASONAM, 2016.
[5] H. Zhang, M. Kan, Y. Liu, S. Ma, “Online Social Network Profile
Linkage,” AIRS, 2014.
[6] S. Liu, S. Wang, F. Zhu, J. Zhang, R. Krishnan, “HYDRA: large-scale
social identity linkage via heterogeneous behavior modeling,” SIGMOD
## Conference, 2014.
[7] W. W.Cohen, J. Richman, “Learning to match and cluster large high-
dimensional data sets for data integration,” KDD, 2002.
[8] A. Malhotra, L. C.Totti, W. Meira Jr., P. Kumaraguru, V. A. F. Almeida,
“Studying User Footprints in Different Online Social Networks,” 2012
IEEE/ACM International Conference on Advances in Social Networks
Analysis and Mining, 2012.
[9] A. Nunes, P. Calado, B. Martins, “Resolving user identities over social
networks through supervised learning and rich similarity features,” SAC,
## 2012.
[10] J. Liu, F. Zhang, X. Song, Y. Song, C. Lin, H. Hon, “What’s in a name?:
an unsupervised approach to link users across communities,” WSDM,
## 2013.
[11] W. E. Winkler, “Overview of record linkage and current research
directions,” BUREAU OF THE CENSUS, 2006.
[12] B. Lim, D. Lu, T. Chen, M. Kan, “# mytweet via instagram: Exploring
user behaviour across multiple social networks,” Advances in Social Net-
works Analysis and Mining (ASONAM), 2015 IEEE/ACM International
Conference on, 2015.
[13] R. Zafarani, H. Liu, “Connecting Corresponding Identities across Com-
munities.,” ICWSM, 2009.
[14] T. Baltruvsaitis, P. Robinson, L. Morency, “Openface: an open source
facial behavior analysis toolkit,” Applications of Computer Vision
(WACV), 2016 IEEE Winter Conference on, 2016.
[15] W. Cohen, P. Ravikumar, S. Fienberg, “A comparison of string metrics
for matching names and records,” Kdd workshop on data cleaning and
object consolidation, 2003.
[16] L. Manikonda, V. V. Meduri, S. Kambhampati, “Tweeting the Mind and
Instagramming the Heart: Exploring Differentiated Content Sharing on
Social Media,” ICWSM, 2016.
[17] F. Benevenuto, T. Rodrigues, M. Cha, V. A. F. Almeida, “Character-
izing user behavior in online social networks,” Internet Measurement
## Conference, 2009.
[18] P. Bhattacharya, M. BilalZafar, N. Ganguly, S. Ghosh, K. P. Gummadi,
“Inferring user interests in the Twitter social network,” RecSys, 2014.
[19] M. Jamali, M. Ester, “A matrix factorization technique with trust
propagation for recommendation in social networks,” RecSys, 2010.
[20] K. Shu, S. Wang, J. Tang, R. Zafarani, H. Liu, “User Identity Linkage
across Online Social Networks: A Review,” SIGKDD, 2016.
[21] J. Klausen, “Tweeting the Jihad: Social media networks of Western
foreign fighters in Syria and Iraq,” Studies in Conflict & Terrorism,
## 2015.
[22] R. Zafarani, H. Liu, “Connecting users across social media sites: a
behavioral-modeling approach,” KDD, 2013.
[23] S. C. Herring, S. Kapidzic, “Teens, Gender, and Self-Presentation in
Social Media,” KDD, 2014.
[24] G. S. O’Keeffe, K. Clarke-Pearson, “The impact of social media on
children, adolescents, and families.,” Pediatrics, 2011.
[25] D. Miller, E. Costa, N. Haynes, T. McDonald, R. Nicolescu, J. Sinanan,
J. Spyer, S. Venkatraman, X. Wang, “How the world changed social
media,” UCL press, 2016.
[26] R. Gross, A. Aless, “Information revelation and privacy in online social
networks,” WPES, 2005.
[27] J. M. Kleinberg, “Challenges in mining social network data: processes,
privacy, and paradoxes,” KDD, 2007.
[28] P. Jain, P. Kumaraguru, A. Joshi, “@I Seek ’Fb.Me’: Identifying Users
Across Multiple Online Social Networks,” WWW, 2013.
[29] A. Goldstein, A. Kapelner, J. Bleich, E. Pitkin, “Peeking inside the black
box: Visualizing statistical learning with plots of individual conditional
expectation,” journal of Computational and Graphical Statistics, 2015.
[30] G. Quercini, N. Bennacer, M. Ghufran, C. NanaJipmo, “LIAISON: rec-
onciLIAtion of Individuals Profiles Across SOcial Networks,” Advances
in Knowledge Discovery and Management: Volume 6, 2017.
[31] Y. Li, Z. Zhang, Y. Peng, H. Yin, Q. Xu, “Matching user accounts based
on user generated content across social networks,” Future Generation
## Computer Systems, 2018.
[32] X. Mu, F. Zhu, E. Lim, J. Xiao, J. Wang, Z. Zhou, “User Identity
Linkage by Latent User Space Modelling,” Proceedings of the 22Nd
ACM SIGKDD International Conference on Knowledge Discovery and
## Data Mining, 2016.
[33] G. Silvestri, J. Yang, A. Bozzon, A. Tagarelli, “Linking Accounts
across Social Networks: the Case of StackOverflow, Github and Twitter,”
KDWeb, 2015.
[34] P. Christen “A Comparison of Personal Name Matching: Techniques and
Practical Issues,” Workshops Proceedings of the 6th IEEE ICDM, 2006.
[35] M. Krieger and D. Ahn “TweetMotif: exploratory search and topic
summarization for Twitter,” In Proc. of AAAI Conference on Weblogs
and Social, 2010
## 267
Authorized licensed use limited to: Utah State University. Downloaded on August 19,2020 at 21:49:58 UTC from IEEE Xplore.  Restrictions apply.