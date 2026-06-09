from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
import json
import re

chat= ChatOllama(
    model="qwen3.5:2b",
    temperature=0,
    reasoning=False
)


def get_foldername(text):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a name giver help assign a name"),
        ("human", """These documents belong to the same semantic cluster.

    Give:
    1. strictly just give one or 2 word category name nothing else
    nothing else

    Documents:
    {text} """)
    ])

    # --- Build the output parser ---
    parser = StrOutputParser()  

   
    parser = StrOutputParser()

  
    from langchain_core.runnables import RunnablePassthrough


    mychain=prompt | chat | parser 
    result = mychain.invoke({"text": text})

    return result
# response = llm.invoke("Explain machine learning simply")

# print(response.content)
if __name__=="__main__":
    ans=get_foldername("Files in group:\n- topic51.txt\n- topic52.txt\n\nRepresentative excerpts:\n[Excerpt 1]: The ability to experience lucid dreams raises profound philosophical questions about the nature of reality, consciousness, and free will. If one can have experiences that feel completely real while knowing they are constructed by the mind, what does this say about waking reality? Philosophers like D\n\n[Excerpt 2]: and agency: if the dreamer can choose actions within a self-generated world, does this reflect genuine freedom or merely another layer of mental simulation? Lucid dreaming blurs the boundary between waking and sleeping states, suggesting that consciousness may exist on a spectrum rather than as a bi\n\n[Excerpt 3]: exploring consciousness itself. Some advanced lucid dreamers claim to experience “false awakenings” or even enter dream worlds that feel more vivid than waking life. Ethical and psychological considerations arise when using lucid dreaming for therapeutic purposes. While it can help with PTSD by allo\n\n[Excerpt 4]: Lucid dreaming occurs when a person becomes aware they are dreaming while still inside the dream, often gaining the ability to control elements of the experience. Research has identified several reliable techniques for increasing the frequency of lucid dreams. Reality checks—habitually questioning w'")
    print(ans)