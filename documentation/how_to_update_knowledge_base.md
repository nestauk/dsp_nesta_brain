# Guide to updating the Nesta Brain knowledge base

[work in progress]


## Introduction

This document is a guide to updating the knowledge base of documents which Nesta Brain uses to answer questions; that is, how to ingest new information into the LanceDB vector database used for retrieval.

Currently, the code is set up to ingest web pages and PDFs from the Nesta website.

The code reflects the data ingestion needs for early prototypes of the Nesta Brain project, written and deployed fairly rapidly, and with limited time for tidying and rationalisation. It may need to be reviewed and simplified for optimal future usability and maintainability, as well as extended to allow ingestion from other sources.


## Relevant code

Most of the relevant code is in `retrieval/db/ingest/nesta_brain.py` and `retrieval/db/ingest/ingest.py`. See also `scraping/scrape.py` and `scraping/scrape_pdf.py` for web and PDF scraping functions.

## Vector database

### DBMS

LanceDB was chosen because it is a free, serverless database which is simple to use and allows a great deal of flexibility in the metadata associated with each vector.

### Schema

The records which the database contains are defined by a schema made up of two Pydantic classes, `Document` and `Chunk` in `retrieval/db/schema/nesta_brain.py`. Note that the metadata associated with each chunk record is contained in the `source` nested field, which represents the `Document` the chunk text is derived from. 

`Document` fields are as follows:

**`location`**: `str`
> the url or file system path where the document can be found

**`title`**: `str`
> document title

**`date_pub`**: `Optional[date]`
> document publication date

*metadata unique to Nesta webpages (derived from the data layer)*

**`projects`**: `Optional[List[str]]`
> the projects the webpage relates to

**`units`**: `Optional[List[str]]`
> the Nesta units the webpage relates to

**`areas_of_work`**: `Optional[List[str]]`
> the Nesta areas of work the webpage relates to

**`missions`**: `Optional[List[str]]`
> the Nesta missions the webpage relates to

**`authors`**: `Optional[List[str]]`
> webpage authors

**`contentType`**: `Optional[str]`
> content type, e.g. person page, unit page, feature page

*traffic metadata*

**`views`**: `Optional[int]`
> no. views

**`rank`**: `Optional[int]`
> webpage rank in terms of views (relative to other webpages on the same site)

*other*

**`drive_type`**: `Optional[str]`
> the type of document on Google Drive (e.g. 'policy' for HR policies)

`Chunk` fields are as follows:

*(inherited from BaseChunk)*

**`text`**: `str`
> the text to be vectorized

**`vector`**: `Vector` (LanceDB class)
> the vectorized text

**`order_index`**: Optional[int] = None
> the order in which the chunk appears in the document it is derived from, relative to other chunks from the same document

**`time_added`**: datetime
> the date and time the chunk was added to the database

*(for chunks explicitly derived from documents fitting the Document class, as in NestaBrain)*

**`source`**: Document
> a nested field representing the document the chunk is derived from

## Data

The database `nesta_brain` contains a vectorized version of the webpages on Nesta's website (as of 29th October 2024), as well as key PDFs from the website (see **PDF scraping** for a description). It also contains 32 HR policies, current as of March 2025. 

There are various methods for ingesting data into the database – see description of the different ingestion 'modes' below.

The main method used for creating the `nesta_brain` database was `web_dump`.

A data dump containing the HTML files and PDFs from the website was created and can be downloaded from the `discovery-iss` bucket on Amazon S3 (`data/nesta_brain/website_2024-10-29.zip`). See also `dsp_nesta_brain/getters/nesta.py` for instructions on downloading it.

Metadata on each webpage is contained in the file `metadata.jsonl`, also included in the data dump. In `web_dump` mode the details of the webpages to scrape are taken from this file and read into a dataframe. Any webpages with a `_status_code` not equal to 200 are removed from the dataframe. The dataframe should therefore only contain webpages which have been successfully downloaded and included in the data dump.

`web_dump` mode is most appropriate when initialising or doing a total refresh of the database. If updating a selection of webpages then `given_urls` mode is more suitable. 

### Webpage scraping

Note that the code in `scraping/scrape.py` is set up to scrape pages from the Nesta website. It contains functions which test whether `div` and `p` elements are desriable based on how Nesta webpages are structured. If scraping pages from other websites, then this code may need to be adjusted accordingly. Alternatively, look at examples of how others scrape generic webapges for RAG systems and/or refer to [LangChain documentation](https://python.langchain.com/v0.1/docs/use_cases/web_scraping/). 


### PDF scraping

As PDFs do not appear on the sitemap, PDFs were identified via links on webpages. Only PDFs which could be assumed to be major reports or other important documents forming the main topic of a webpage were ingested (see `download_button_pdf_only` option below for an explanation).

PDFs were read using the [Unstructured](https://unstructured.io/) open source library. This attempts to recognise different components of a PDF, for example, title, headers and footers, narrative text, etc. Many of the PDFs on Nesta's website are mostly diagrammatic rather than conventional reports, which makes it harder to identify the kind of narrative text we actually want to ingest accurately in an automated fashion. Attempts were made however to automatically identify and remove extraneous text, for example, title pages, tables of contents, reference lists, headers and footers, and so on, before ingestion. See code in `scraping/pdf`.

## Embeddings model

OpenAI's `text-embedding-3-small` model was used for `nesta_brain` and `full_site_demo_db_with_pdfs`. Future users may want to experiment with different embeddings models. The default embeddings model can be set in `config.py` via `DEFAULT_EMBEDDINGS_MODEL`, and the relevant code is in `retrieval/embeddings.py`.

## Ingestion modes and settings

Most of the code for ingest text sources into the database is in `retrieval/db/ingest/nesta_brain.py`.

There are five 'modes' for ingestion depending on the source and structure of the data to be ingested. The modes are designed to ingest the following content:

**`web_dump`**: webpages/PDFs which are included in a data dump of the Nesta website and are contained in directories with paths `WEBSITE_DATA_PATH` or `PDF_PATH` 

**`web_search`**: webpages resulting from a Google programmable search 

**`given_urls`**: webpages derived from a list of urls supplied by the user 

**`from_csv`**: data contained in a CSV file

**`from_drive`**: documents stored in a Google Drive accessed via a specified service account. See `google_api`drive.py` for details.

The mode is specified via the command line, as follows:

**`-m`: mode**:
> `wd`, `ws`, `csv` or `drv` for `web_dump`,`web_search`,`from_csv` or `from_drive` mode respectively.

If you require `given_urls` mode, then use:
**`--urls`: urls**:
> a list of space-separated urls to ingest

Depending on the mode, additional arguments may also need to be passed on via the command line, as follows.

*universal arguments*

**`-r` : replace flag**: 
> if the replace flag is present, if a document already exists in the DB, any chunks previously derived from it will be deleted and replaced. If it is absent, previously existing chunks will be left but new chunks will not be added. In both cases, chunks are not added to the database if any chunks have been previously added from the same document, using each document's `location` as a unique identifier. This is to avoid duplication of chunks.

*arguments only relevant to `web_dump` mode*

**`--pdf` : PDF flag**: 
> if present, look for and ingest PDFs downloaded from the website and stored in the directory with path `PDF_PATH`; if absent, look for and ingest downloaded webpages stored in the directory with path `WEBSITE_DATA_PATH`

**`--all` : all PDFs flag**:
> A problem encountered while ingesting PDFs was detecting and inserting their metadata (title, publication date). We found this could not be reliably automatically identified from the PDF itself. For this reason, where a webpage has obviously been created to promote a new report, it is simplest to assume the PDF has the same metadata as the webpage. If `--all` is absent, then only those PDFs linked to by a red 'Download' button on the originating webpage are ingested; if `present`, all PDFs linked to by the originating webpage are looked for and ingested, as long as they are on the Nesta website. **This is not recommended.** Note that if the cautious flag (see below) is absent, then all PDFs are given the same title and publication date as the originating webpage. This could result in a significant number of PDFs in the database with inaccurate metadata.

**`-c` : cautious flag**: `bool`
> if `True` the system will pause to check whether the user wants to scrape every individual PDF in turn, and will also ask whether metadata guesses are correct. If `False`, the system will scrape each PDF automatically and assume the metadata from the originating webpage is correct, as described above. Setting `cautious` to True is very slow and only intended for testing, or for getting a feel for the available PDFs, or for ingesting a small number of PDFs.

**`start_index`**: `int`
> the row of the metadata dataframe (see **Data** section) to start ingesting from, with the first row at index `0`. `start_index` is taken from the first command line argument, or defaults to `0`.

**`batch_size`**: `int`
> the number of webpages or PDFs to get embeddings for and ingest at a time. Note that if `batch_size` is too high then you will get error messages back from OpenAI (see **Known issues**). Users are encouraged to experiment with `batch_size`. PDFs can be large and slow to scrape, so a very low batch_size (<5) is recommended if `pdf_mode` is `True`. A `batch_size` of 50 for webpages and 1 for PDFs was used when the DB was originally set up. Batch sizes > 100 for webpages seemed to cause problems. (Note that this in document when the word 'batch' is used this is not with reference to OpenAI's Batch API, which is not used.)

*Settings relevant to `web_search` mode*

**`query`**: `str`
> the web search query (as if doing a Google search)

**`site_url`**: `str`
> the url of the website which is the target of the search

**`subdirectories`**:`Optional[List[str]]`
> a list of subdirectories on the website which you wish to limit the search to; note that a separate search will be conducted for each of these subdirectories in turn

*Google Programmable Search limits*: Note that only a 100 search results can be returned from Google Programmable Search for each distinct search, where a distinct search is a combination of query, url and subdirectory. Furthermore, if more than 100 searches a day are required, a billing account will need to be set up. See [this Google webpage](https://developers.google.com/custom-search/v1/overview#:~:text=Custom%20Search%20JSON%20API%20provides,to%2010k%20queries%20per%20day.) for more details.

*Settings relevant to `given_urls` mode*

**`given_urls`**: `List[str]`
> a specified list of urls pointing to webpages to scrape and ingest

## Adding new data sets

As the website changes over time, the vector database underlying Nesta Brain will need to reflect this. In addition, admins may wish to add pages from other sites, or offline documents.

### Updates to the Nesta website: adding new pages in `given_urls` mode

If it is necessary to ingest only a relatively small number of pages which have been added to the Nesta website since October 2024, then the following step can be taken:

1. If the urls of the webpages are known, then they can be added by setting `mode` to `'given_urls'` and setting the `given_urls` variable to the list of urls.

Note that if `mode == "given_urls"` scraping of Nesta webpages hould automatically yield the full range of metadata contained in the data layer on each page (although some of the fields are often left blank).

### Updates to the Nesta website: `web_dump` mode

`web_dump` mode can be used either to add new pages or to do a completely new reingestion of the whole site.

[an explanation of how to derive urls from the site map, download them and add their metadata to `metadata.jsonl` needs to be added here.]

New pages can be added in `web_dump` mode. Only webpages which are not already in the database are added. The code will iterate through the rows of the dataframe representing all webpages, and those already present will be ignored. If any PDFs linked to by new pages are wanted as well, then run the code twice, once with `pdf_mode = False` (to ingest the webpages), and once with `pdf_mode = True` (to ingest the PDFs).

For a completely new reingestion, a new database needs to be created, following these steps:

1. Change DB_PATH in config.py to the path of the new database.
2. Run `__main__` in `retrieval/db/schema/nesta_brain.py` to set up the new database.
3. Run `__main__` in `retrieval/db/ingest/nesta_brain.py` setting `mode = "web_dump"`. Again, if PDFs are wanted as well, then run twice, once with `pdf_mode = False`, and once with `pdf_mode = True`. Note that ingesting the entire site can take many hours and may throw the occasional error. If errors are encountered, investigate and fix the issue, or use `start_index` to skip the webpage which caused the error to be thrown.

### Adding webpages from other sites

Webpages from other sites can potentially be added, either in `web_search` or `given_urls` mode. For example, to scrape pages on Medium by the Nesta data science team follow these steps:

1. Set `mode = "web_search"`.
2. Set `site_url = "https://medium.com/data-analytics-at-nesta"`.
3. If looking for a particular topic, set `query` to look for related keywords or phrases. Alternatively, set `query` equal to the empty string or `query = "-site:https://medium.com/data-analytics-at-nesta/tagged"` to exclude particular subdirectories.
4. Set `subdirectories` equal to a list of any desired subdirectories, or set equal to `None` or `[]`.
5. Run `__main__` in `retrieval/db/ingest/nesta_brain.py`.

Note that the code which does the website scraping in `scraping/scrape.py` is designed for Nesta webpages and may need some editing to be suitable for other websites, for example, in deriving the publication date, or determining which page elements count as text and which you wish to ignore. See the `scrape` and `html_to_text` functions in `scraping/scrape.py`.

### Adding offline resources

The code in `retrieval/db/ingest/nesta_brain.py` is not currently set up to ingest offline resources (other than webpages and PDFs stored during a data dump of the Nesta website and accessed in `web_dump` mode). Code would need to be written to convert the offline documents into plain text, and then convert the plain text and any metadata into the LangChain [`Document`](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) class. A list of LangChain `Document`s can be passed to the `ingest` function via the `documents` argument. The `ingest` function chunks and vectorizes each document and inserts the chunks into the vector database. Note that every `Document` in `documents` must have a `location` (system file path or url) and `title` field in its metadata – all other metadata is optional. `location` is used as a unique identifier for documents to establish whether documents have already been added to the database. \[Note that the LangChain `Document` class is distinct from the `Document` class defined in the DB schema. The aliases `LangchainDocument` and `LanceDocument` are used for these classes respectively in `retrieval/db/ingest/nesta_brain.py` to avoid confusion].

## Known issues

1. A throttle should (theoretically) ensure OpenAI requests are kept within rate limits. However, when `batch_size` is large error messages can be thrown by the API which are not due to rate limits being exceeded, or by the lancedb package. Accordingly, users may find that the rate limits are not in danger of being breached because the batch sizes need to be relatively small to avoid these latter errors. There wasn't time to troubleshoot and fix these issues, but future users should be aware that if they wish to ingest large volumes of documents simultaneously, they may need to investigate the causes of these errors and upgrade the code.
2. There were some PDFs which didn't scrape successfully and which threw error messages, probably due to size. There also wasn't time to investigate and fix this. Future users may encounter the same problem. If a PDF throws an error, it can be skipped by noting the row in the metadata dataframe of the originating webpage and setting `start_index` to the one following it.
3. As mentioned above, the `location` metadata field is used a as a unique identifier for documents to avoid duplicate scraping of webpages and other documents. However, the database does currently contain some duplication of webpages where there are URL aliases in the site map. These should be removed from the database, time-permitting, and code added to retrieval/db/ingest/nesta_brain.py to prevent this occurring.
