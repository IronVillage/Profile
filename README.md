## Andrew (Drew) Millard

*Andrewmillard123@gmail.com*

## Applied Quantitative Modeler | Sports & Esports Markets

-----

I'm a quantitative modeler with a five-year record of building profitable models for Esports and UFC markets. My process is rooted in first-principles domain analysis, engineering high-impact features and the strategic use of model ensembles. My current focus is the player prop market (PrizePicks, etc.), where I am architecting agent-based data pipelines to fuel advanced machine learning applications and my private modeling.

-----

## My Modeling Approach

I approach modeling with a "forever student" mindset, believing there is always value in exploration. However, through years of practical application and studying the processes of other quants, I've learned a clear hierarchy governs what truly improves a model. This is a high level look at how I approach a problem.

### Data

This is always where you have to start, however I often find myself revisiting if I am really hitting a wall. My largest jumps in performance always come from a new data source, which is why I place the highest importance on acquiring clean and diverse data. Garbage in, garbage out.

### Feature Engineering and Domain Knowledge

Inspired by Sabermetrics and analysts like Ken Pomeroy, traditional statistics and truly making an effort to understand the domain I am working in plays a huge role in the success of my models. I excel at transforming deep, practical domain analysis into novel, high-impact features.

### Advanced Modeling Architecture

I am experienced in applying all ensemble methods, stacking, blending, bagging etc.., using them in nearly all of my models to build increasingly complex and powerful architectures. I've always found it to be a bit of an art making these and really enjoy working with them even if you find out they are not always needed. Sometimes a shallow simple model with heavy feature engineering is all you need.

-----

## CSGO/CS2 Match Winner Model

**Objective:** To consistently identify profitable betting opportunities by developing a predictive model that outperforms the market for pre-match CSGO odds.

### Model

After extensive experimentation, trying to parse demos using tons of features, my biggest breakthroughs happened when I created powerful novel statistics and adjusted them for strength of schedule.

### Key Innovations & Components

  * novel efficiency metric, `kills per $20k starting equipment`, to more accurately measure a team's offensive output relative to their economic investment, outperformed kills per round and rounds won.
  * powerful team rating metrics, although not novel by any means, using an elo rating with various k values, bagged, outperformed any single k value elo rating. Now I use Glicko, WHR or soon my own implementation of TrueSkill 2.
  * adjusting for strength-of-schedule, primarily using a Massey rating system to ensure performance against top-tier opponents was valued appropriately.
  * capture of multi-dimensional view of team form, features were generated across various time windows and ELO K-values, modeling both short-term momentum and long-term skill.
  * The model utilized a two-stage stacking architecture, generating out-of-fold (OOF) predictions from each feature set, boosting the final model's performance significantly.

### Performance Metric

The model's performance was measured by a single, practical metric: **profitability (ROI)** against the average market price 24 hours prior to start. I however was always looking at log loss, accuracy, max draw down of bank roll.

### Possible Improvements

Decided not to work with round by round data here, so many more possibilities for features and different models. Monte Carlo sims etc.. I am working with round data now in my current work, and plan on revisiting CS2 soon and upgrading it seriously.

### Model Architecture

```
* All bagging done was 80% sample data & 80% features.

       Feature Set A             Feature Set B            Feature Set C
   ┌────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
   │  massey_rating_30  │    │  elo_rating_k13  │    │  adj_rounds_won_6m  │
   │  massey_rating_25  │    │  elo_rating_k21  │    │  adj_rounds_won_5m  │
   │  massey_rating_20  │    │  elo_rating_k34  │    │  adj_rounds_won_4m  │
   │  massey_rating_15  │    │  elo_rating_k55  │    │  adj_rounds_won_3m  │
   │  massey_rating_10  │    │  elo_rating_k89  │    │  adj_rounds_won_2m  │
   │  massey_rating_05  │    │  elo_rating_k144 │    │  adj_rounds_won_1m  │
   └─────────┬──────────┘    └────────┬─────────┘    └──────────┬──────────┘
             │                        │                         │
             ▼                        ▼                         ▼
     [CV OOF Bagging CF]      [CV OOF Bagging CF]      [CV OOF Bagging CF]
             │                        │                         │
             └────────────────────────┼─────────────────────────┘
                                      │
                            ┌─────────┴──────────┐
                            │  massey_oof_preds  │
                            │   elo_oof_preds    │
                            │   adj_oof_preds    │
                            └─────────┬──────────┘
                                      │
                                      ▼
                              [Meta-Model Bagging]
                                      │
                                      ▼
                            [Final Win Probability]
```

-----

## Current Projects: Crowdslips

As I am gearing up again, I have been building some special data feeds for my model. One of those feeds is a sentiment and bet data feed. These are real bets being taken from social media, either as bet slips posted, or mentions in chats. If you want to ask if I get data from *somewhere* chances are I do or will.

Since I did not really know of any sites that offered these stats, I figured I'd build a front end, and create real sentiment data site. Its in the early stages, but I think there are a ton of possibilities with this.

Very simple structure, will not get into the real guts of it here, but always happy to talk about it :). I sent you a special link, with some features I am thinking about.


```
[Social Media Sentiment Data]────────────────────────────┐
                                                         ├─>[AI Parser]──>[Aggregated Database]──>[Modeling]
[Socials Betting Data ]────>[DFS Bet Slip Classifier]────┘
```

-----

## Current Projects: Chance May Crown

This is a concept for a private, limited-membership syndicate focused on providing model-driven entries for DFS platforms like PrizePicks.

The sell is the 'bet engine' designed for dynamic risk and portfolio management. It solves the complex sizing and allocation problems that arise when managing multiple, partially-live parlay entries, adjusting for shifting probabilities in real-time.

Very excited about this one, will talk about it or answer any questions. :)

-----

## Future Focus & Outlook

I think this is a great time to be modeling. Human behavior is what moves markets, and its never been easier to collect that data if you can make sense of it.

My current independent research is focused on developing agentic data pipelines—automated systems designed to source and structure unique, real-time data that isn't currently being captured by traditional methods. Ask me about it! I have some crazy ones I'd love to talk about.

The prospect of applying these concepts in a collaborative environment is incredibly exciting to me. The opportunity to work with a talented team on the complex, large-scale modeling challenges at a company like Oddin.gg is exactly the kind of environment where my skills would thrive.

---
## Technical Stack 

* **Primary Language:** Python
* **Data Science & ML:** Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, Vertex AI
* **Data Layer:** Firestore & BigQuery
* **Cloud & DevOps:** Google Cloud Platform (GCP), Docker, Cloud Run, Cloud Functions
* **Web Scraping & Automation:** Scrapy, Beautiful Soup, Requests, SeleniumBase, Apify Client
---

## Professional Journey

**2015 - 2019**
* First sports bet and then down the rabbit hole... Began self education in statistical analysis and theory, behavioral economics, anything related to decision making, prediction, forecasting. (Thinking Fast, Thinking Slow etc...)
* Transitioned to full-time focus on poker, sports betting, and crypto. Specialized in UFC markets for betting.


**2020 - 2024**
* In response to the flash crash and COVID, shifted focused entirely to algorithmic modeling, esports was of course obvious choice.
* Learned Python, Pandas, and machine learning fundamentals through self-study with books and online courses, but mainly through actually building things.
* After lots of trial and error, numerous failures and false starts, beagn to build profitable models. 


**2024 - Present**
* Shifted focus from pure modeling to building data platforms.
* Began development on two platforms: CrowdSlips and Chance May Crown.
* I am now diving back into modeling with a new love, focusing on prizepicks player props as a test case for my new methods and data feeds.

## Code Sample



## Slips Classifier



This is one of the more recent things I have done. It is not predictive modeling, but I think it is a solid representation of how I problem solve. I needed to not waste money sending garbage images, gifs, and more importantly finished bet slips to my vertex parsing model, so I build this image classifier, that has an almost 98% I think accuracy with deciding if a DFS bet slip has an upcoming match to parse.




Full disclosure I use cursor to code now, but the architecture, model choice and parameter tuning is all me. The solution to my problems are mine, I use AI as a tool to speed up large scripts, where I am absolutely sure what I want. I think about it as if I was working with a partner developer.



Also when I work with actual data for modeling, EDA, feature engineering, etc... I do that all myself in a notebook, as I can iterate and test much faster than an AI can I feel.



Happy to answer any questions about this or anything else! Looking forward to speaking with you.
