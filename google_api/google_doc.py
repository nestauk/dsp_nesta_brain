from typing import Optional


class GoogleDoc(dict):
    """Class to manipulate the contents of Google Docs returned from Drive"""

    text_: Optional[str] = None

    @property
    def id(self) -> str:
        """Get the document ID"""
        return self.get("documentId")

    @property
    def text(self) -> str:
        """Convert contents into a single text string if this has not already been done; return the resulting text"""

        if self.text_ is None:

            self.text_ = ""
            for element in self["body"]["content"]:

                if (
                    element.get("paragraph")
                    and element["paragraph"]["paragraphStyle"].get("namedStyleType") == "NORMAL_TEXT"
                ):
                    self.text_ += element["paragraph"]["elements"][0]["textRun"]["content"]

        return self.text_

    @property
    def title(self) -> str:
        """Get the document title"""
        return self.get("title")
