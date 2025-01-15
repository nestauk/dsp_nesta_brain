from typing import Optional


class GoogleDoc(dict):
    """Class to manipulate the contents of Google Docs returned from Drive"""

    text_: Optional[str] = None

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
