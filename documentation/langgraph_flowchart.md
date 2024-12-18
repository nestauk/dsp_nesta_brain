# LangGraph Flowchart

Use of LangGraph is currently very simple and limited, with only two nodes.

This graph is appended prior to the retriever.

The state represents input to the retriever.

After the user types in their input, the nodes decide whether filter conditions need to be set for retrieval based on the user's input.

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([<p>__start__</p>]):::first
	decide_if_person_page(decide_if_person_page)
	decide_if_need_time_constraint(decide_if_need_time_constraint)
	__end__([<p>__end__</p>]):::last
	__start__ --> decide_if_person_page;
	decide_if_need_time_constraint --> __end__;
	decide_if_person_page --> decide_if_need_time_constraint;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
