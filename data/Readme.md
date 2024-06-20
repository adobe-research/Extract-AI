## About the Data

To create the Coherent Extractive Summarization Dataset (CESD), we have carefully curated a diverse collection of 1000 instances spanning five distinct categories: (1) News, (2) Debate, (3) TV Show, (4) Meeting, and (5) Dialogue. Within CESD, each category is represented by precisely 200 instances, selected through random sampling from the following datasets:

| Category | Dataset | Link
| ------------- | ------------- | ------------ |
| News | CNN/DM | https://github.com/abisee/cnn-dailymail
| Debate | DebateSum | https://huggingface.co/datasets/Hellisotherpeople/DebateSum
| TV Show | TVR | https://github.com/jayleicn/TVRetrieval/tree/master
| Meeting | MeetingBank | https://meetingbank.github.io/
| Dialogue | DIALOGSum | https://huggingface.co/datasets/knkarthick/dialogsum

You can easily acquire each dataset by accessing the provided link. Once you have successfully downloaded the data, you can proceed to utilize the `src/sampling_data.py` script to perform random sampling on each dataset. 

Upon running this script, you will obtain a set of output files in the form of five .csv files. To make it easy for the user, we have included these files in the `/data/sampled_data` folder. Nevertheless, this script can be a valuable pathway for including a new dataset for processing (You can simply add a new dataset processing function).

**Note**: <mark>We have released the main source document content for the Debate, Meeting, and Dialogue categories. However, due to licensing issues, the content for News and TV Shows has not been made available.</mark>


### Model Summary Generation using Falcon-40B-Instruct

For the annotation, the objective was two-fold: (1) to create a coherent summary extracting important sentences/phrases from a document that effectively captures the key aspects of the document, and (2) to provide feedback on the steps to go from the model summary to the gold summary. In this process, annotators are provided text document and model summary.

You can use `/data/sampled_data` to acquire text documents across five categories. For generating a model summary, we use [Falcon-40B-Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct). We provide a script to generate model summaries that are used in the data annotation process. You can find the script at path `/data/src/generate_model_summary.ipynb`. Please refer to the script for more details.

In this script, just provide the path of `/data/sampled_data` and this script will generate model summaries for you. To ensure the extraction of meaningful sentences for the model summary, and recognize Falcon's limitations in accurate text generation, we employ the following prompt strategy. Instead of instructing Falcon to generate sentences, we prompt the model to identify and select the most coherent sentences from the document in order to create the summary.

Prompt:
```
You are an extractive summarizer. You are presented with a document. The document is a collection of sentences and each sentence is numbered with sentence ids. Understand the given document and create a meaningful summary by picking sentences from the document. Please list the sentence IDs as output so that sentences corresponding to the generated IDs summarize the document coherently. Learn from the below example:

Document:
1. In "Rules about Copying and Sharing Java Code," author Josh Smith believes that code copied from others should be cited as such, otherwise it is plagiarism.
2. Another important idea that Smith discusses is that most discussions of plagiarism are with respect to "works in written and spoken language," and hence he wants to discuss how to cite the work of others within computer programs.
3. He supports this latter idea by specifying that "due credit" is given to others by specifying the original author, the source where the code was obtained, and any alterations that the current author is making to the original code.
4. The author provides examples of citations whose source is from a textbook, an instructor, the Internet, from multiple sources, and from code that is "common knowledge" in order to show how one can always clearly identify the author of each code unit in a variety of situations.
5. Another important point made by Smith is that code should never be transferred between students electronically, because this would imply unsuitable sharing of work and plagiarism.
6. Smith's target audience is computer science students, as it is likely that either they are unaware of plagiarism in general, or they are aware of plagiarism in other fields but have not considered how it applies specifically when writing code.
7. This material relates to the current course material because it comes after the design process and during the implementation process, when the most code is being written and would be most available for potential copying.
8. Smith's guidelines for copying and reusing code are accurate and useful; however, he forgets that sometimes a great deal can be learned by examining code written by others.
9. It would have been nice if he had left some provision where it was okay to do this under the right circumstances.

Summary: <s> [1, 2, 5, 6, 8]

Document: [input text document]

Please Create a concise summary using as few sentences as possible.

Summary: <s>
```

In this context, the input document is provided in sentence format to facilitate the process of coherent sentence selection for the summary using Falcon.
