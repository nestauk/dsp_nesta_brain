# Guide to updating the Nesta Brain knowledge base


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
> the url or file system path where the document can be found (unique identifier)

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

**`-c` : cautious flag**:
> if present, the system will pause to check whether the user wants to scrape every individual PDF in turn, and will also ask whether metadata guesses are correct. If absent, PDFs are scraped automatically and it is assumed that the metadata from the originating webpage is correct, as described above. Setting the cautious flag is very slow and only intended for testing, or for getting a feel for the available PDFs, or for ingesting a small number of PDFs.

*arguments only relevant to `web_dump` and `from_csv` mode*

**`--start_index` : start index**
> an integer which represents the index of a set of data records to start ingesting from. If absent, then the start index defaults to zero. If in `web_dump` mode, start_index represents the row of the metadata dataframe (see **Data** section) to start ingesting from. If in `from_csv` mode, then it represents the row of the dataframe into which the CSV data has been read.

**`--batch_size` : batch size**
> the number of data records (representing webpages, etc.) to get embeddings for and ingest at a time. If absent, then it defaults to 10. Users are strongly encouraged to use much higher batch sizes when ingesting web pages (up to 350 has successfully been tested). The @retry decorator is used to ensure embeddings requests are retried if rate limit or API connection errors are thrown. PDFs can be large and slow to scrape, so a very low batch_size (<5) is recommended. (As a clarification, note that the word 'batch' in this context has no relation to OpenAI's Batch API, which is not used.)

*arguments only relevant to `web_search` mode*

**`--query` : query**
> the web search query (as if doing a Google search)

**`--site` : site url**
> the url of the website which is the target of the search (defaults to Nesta's website)

**`--use_subdirectories` : use subdirectories flag**
> if present and searching Nesta's website, search various subdirectories in turn (see `subdirectories` variable)

*Google Programmable Search credentials and limits*: Users will need Google Programmable search credentials to use `web_search` mode and add the `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ID` variables to their `.env` file. Note that only a 100 search results can be returned from Google Programmable Search for each distinct search, where a distinct search is a combination of query, url and subdirectory. Furthermore, if more than 100 searches a day are required, a billing account will need to be set up. See [this Google webpage](https://developers.google.com/custom-search/v1/overview#:~:text=Custom%20Search%20JSON%20API%20provides,to%2010k%20queries%20per%20day.) for more details.

*arguments only relevant to `given_urls` and `from_drive` mode*

**`--urls`: urls**:
> as mentioned above. When used in `from_drive` mode, it specifies a list of files on Google Drive. These must be accessible via the service account (see `google_api/drive.py`).

*arguments only relevant to `from_drive` mode*

**`--drive-type` : drive type**:
> an optional string which describes the type of drive document which is stored with the document metadata. Only use this if there is a reason somewhere else in the code. Note that the string must be one of the values specified by DriveTypeEnum (you can add to this as you wish).

**`--file_ids` : file IDs**:
> a list of file IDs to ingest (space-separated)

**`--all_drive` : all Drive files flag**:
> if present, look for all accessible files on Google Drive and ingest them all. If absent, you must specify which files you want to ingest via `--file_ids` or `--urls`.


## Adding new data sets: Examples

As the website changes over time, the vector database underlying Nesta Brain will need to reflect this. In addition, admins may wish to add pages from other sites, or offline documents.

### Updates to the Nesta website: adding new pages in `given_urls` mode

If it is necessary to ingest only a relatively small number of pages which have been added to the Nesta website since October 2024, then the following step can be taken:

1. If the urls of the webpages are known, then they can be added in `give_urls` mode via the `--urls` command line argument.

Note that in `given_urls` mode scraping of Nesta webpages should automatically yield the full range of metadata contained in the data layer on each page (although some of the fields are often left blank).

### Updates to the Nesta website: `web_dump` mode

`web_dump` mode can be used either to add new pages or to do a completely new reingestion of the whole site.

You will need to download the relevant webpages and/or PDFs and put them in the directories with paths `WEBSITE_DATA_PATH` and/or `PDF_PATH`. You will also need to create a metadata file equivalent to `scraping/data/website_2024-10-29/metadata.jsonl`.

New pages can be added in `web_dump` mode - only webpages which are not already in the database should be added. The code will iterate through the rows of the dataframe representing all webpages, and those already present will be ignored. If any PDFs linked to by new pages are wanted as well, then run the code twice, once with `--pdf` and once without.

For a completely new reingestion, a new database needs to be created, following these steps:

1. Change DB_PATH in config.py to the path of the new database.
2. Put the following code in a file and run it:
    ```
    import lancedb
    from config import DB_PATH
    from your_schema_module import YourSchemaChunkClass  #import a class representing the DB chunk from wherever you have put it
    #for example, from retrieval.db.schema.nesta_brain import Chunk

    db = lancedb.connect(DB_PATH)
    table = db.create_table("your_chunk_table_name", schema=YourSchemaChunkClass)  #the chunk table in NestaBrain is just called "chunk
    table.create_fts_index("text")  #assuming your chunk class has a text field
    ```
3. Run `__main__` in `retrieval/db/ingest/nesta_brain.py` with `-m wd` on the command line. Again, if PDFs are wanted as well, then run twice, once with `--pdf`, and once without. Note that ingesting the entire site can take a long time and may throw the occasional error. If errors are encountered, investigate and fix the issue, or use `start_index` to skip the webpage which caused the error to be thrown.

### Adding webpages from other sites

Webpages from other sites can potentially be added, either in `web_search` or `given_urls` mode. For example, to scrape pages on Medium by the Nesta data science team follow these steps:

1. Run `__main__` in `retrieval/db/ingest/nesta_brain.py` with `-m ws --site https://medium.com/data-analytics-at-nesta` as command line arguments.

Options:

• If looking for a particular topic, also use the `--query` command line argument to look for related keywords or phrases.

• To look only in particular subdirectories, set the `subdirectories` variable in `retrieval/db/ingest/nesta_brain.py` equal to a list of any desired
subdirectories, and use the `--use_subdirectories` command line argument


### Creating schemas and tables for other chunk-like data

The `BaseChunk` class can be extended to create other tables containing any data which users would like to subject to semantic search. This could include, for example, data which comes from tables with one text-based field which is suitable for semantic search. During the Nesta Brain project, we experimented with ingesting data which did not come from documents but from tables, for example, a table containing all of Nesta's mission projects, and a table of International Aid Transparency Initiative data (see the `MissionProject` class in `retrieval/db/schema/nesta_brain.py` and the `Activity` class in `retrieval/db/schema/policy_atlas.py`). The only constraints on chunk-like data are that it must have a text field and a vector field. The rest of the fields are essentially metadata. The user can specify the field to be used as the text field in the `__init__` function of the chunk class, or simply have a field already labelled 'text' in the input data. See the `chunk_to_Chunk` function in `retrieval/db/ingest/ingest.py` to see where vectorization currently happends in the code base. 

• To look only in particular subdirectories, set the `subdirectories` variable in `retrieval/db/ingest/nesta_brain.py` equal to a list of any desired
subdirectories, and use the `--use_subdirectories` command line argument

Note that the code which does the website scraping in `scraping/scrape.py` is designed for Nesta webpages and may need some editing to be suitable for other websites, for example, in deriving the publication date, or determining which page elements count as text and which you wish to ignore. See the `scrape` and `html_to_text` functions in `scraping/scrape.py`.



## Known issues

1. There were some PDFs which didn't scrape successfully and which threw error messages, probably due to size. There also wasn't time to investigate and fix this. Future users may encounter the same problem. If a PDF throws an error, it can be skipped by noting the row in the metadata dataframe of the originating webpage and setting `start_index` to the one following it.
2. As mentioned above, the `location` metadata field is used as a unique identifier for documents to avoid duplicate scraping of webpages and other documents. However, the database does currently contain some duplication of webpages where there are URL aliases in the site map. These should be removed from the database, time-permitting, and code added to `retrieval/db/ingest/nesta_brain.py` to prevent this occurring.
