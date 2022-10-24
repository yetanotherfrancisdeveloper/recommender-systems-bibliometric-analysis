# Bibliometric analysis of recommender systems
![GitHub](https://img.shields.io/github/license/Wilscos/recommender-systems-bibliometric-analysis)

This is the code used in the paper 'A survey on Neural Recommender Systems:
insights from a bibliographic analysis' (link will be provided when available). In order 
to run the code, you need to install the libraries in the file `requirements.txt` in 
your virtual environment:

`pip install -r requirements.txt`

The results will be saved in the `data` directory. In the same directory you will also 
find the data that we used to perform the analysis.

## How to run

In order to run the script you need to just run the `main.py` script in your environment
or wherever you wish to run it. For instance, in the directory where is the script:

`python main.py`

Or if you are using `pipenv`:

`pipenv run main.py`

You can also specify two arguments:

- `-skip_data_viz`: The default code behaviour is to run a part of the code that produces some nice plots of the data and saves them in the 'plots' directory. If this argument is specified, it will not. You don't need to pass any additional parameter to this argument.
- `-clustering_model`: the default value is 'lda'. You can also opt for 'keybert' and 'dbscan' to obtain the results from these models instead.

### Example

`pipenv run main.py -skip_data_viz -clustering_model keybert`

In this way you are making the plots that are going to be saved in the 'plots' directory, and you are using keybert to get the keywords for each cluster.

`pipenv run main.py -clustering_model dbscan`

In this case you are not making plots of the data, and you are using DBScan to cluster the papers.

`pipenv run main.py`

And if you run the code like this, you won't make the plots and you will use Latent Dirichlet Allocation (LDA) to cluster the papers.

## Some of our results

We scraped all the papers published in open access in the last 5 years that treated 
recommender systems. The number of publications per year can be visualized in the figure 
below.

<p align="center">
    <img alt="Publications per year" src="results\bar_plot_publications_per_year_2017-2021.png" style="height:300px">
</p>

It is easy to notice the steady growth in the number of publications per year. We wanted 
to understand what were the main topics that drove this growth. To do this we decided 
to cluster the last 1000 papers published with the LDA 
algorithm. The results from the algorithm are shown in the image below.

<p align="center">
    <img alt="LDA clustering results" src="results\t-SNE_LDA_topics=5,min_df=0.05,max_df=0.95.png" style="height:300px">
</p>

In this way we were able to find some topics (clusters) of interest that we summarized in 
the following table:

| Topic |                  Label                  |                                                                   Top-k words                                                                    |
|:-----:|:---------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|
|   0   |        Probabilistic approaches         |                       condition, determine, variance,<br/>minimum, estimate, bound, error,<br/>standard, linear, parameter                       |
|   1   |          Graph Neural Networks          |                    capture, feature, batch, attention,<br/>dimension, representation, prediction, <br/>vector, matrix, layer                     |
|   2   | Computer Vision and Data Visualization  |                   classification, label, machine,<br/>description, content, identify, accuracy,<br/>domain, extract, language                    |   
|   3   |                  N.A.                   | effect, application, event, documentation,<br/>technology, control, communication,<br/>access, management, development,<br/>environment, support | 
|   4   |               AI Fairness               |                    reward, reinforce, question, agent,<br/>person, prefer, experience, world,<br/>policy, feedback, decision                     |


The topics were identified on the basis of the most frequent words in each cluster. We also 
checked the papers in order to understand if the words were a good indication and if there 
was a topic that described all the papers in the cluster. The only one that we did not feel 
confident enough to label was the fourth topic, where the papers did not even seem to be 
on recommender systems, but rather on some sort of recommendation made from a different 
type of analysis. If you have any idea about it, let us know by opening an issue or by 
contacting us!

## About the scraper

We got the .pdf files gathered in the [articles_pdf directory](data/articles_pdf) from 
arXiv through the scraper that you can find [here](scraper/arxiv_scraping.py). It is not 
possible tough to get all .pdf files from arxiv for two reasons:

- They interrupt the connection with you after scraping for a while. You would need to 
  change your IP dynamically in order to keep scraping;
- Not all papers uploaded a .pdf file to arXiv, but this could be solved by scraping from 
  Ar5iv, maybe.

While there could be solutions to these two problems, we think it would be more useful to 
create a new library to find the .pdf file of a paper online, if there exists one. We may 
work on this in the future.

## About the data

In the [data directory](data) you will find two directories and a .csv file with the 
necessary data to run the scripts and perform the analysis:

- [.pdf articles](data/articles_pdf)
- [.txt articles](data/texts)
- [.csv file](data/df.csv)

To convert the .pdf files to .txt files we slightly modified 
[this repository](https://github.com/yuchangfeng/PDFBoT) from
[Extracting Body Text from Academic PDF Documents for Text Mining (Yu et al., 2020)](https://arxiv.org/pdf/2010.12647.pdf). 
The changes made simply allowed us to use the repository, but the main application was 
the one that you can find in the repository. We may share in the future what we did in 
order to make it work easily.
