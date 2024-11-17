import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
import ollama


class Website:
    """
    A utility class to represent a Website that we have scraped
    """

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url):
    website = Website(url)
    response = ollama.chat(model=MODEL, messages=messages_for(website))
    return response['message']['content']

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

display_summary("https://en.wikipedia.org/wiki/Machine_learning")


# The response generated is as follows

# Machine learning is a field of computer science that involves the use of algorithms and statistical models to enable computers to learn from data, without being explicitly programmed. This type of machine learning has numerous applications in various industries such as business, healthcare, finance, and technology.

# Types of Machine Learning

# Supervised Learning: In this type of learning, the algorithm is trained on labeled data, where the correct output is already known.
# Unsupervised Learning: In this type of learning, the algorithm is trained on unlabeled data, and it must find patterns or relationships in the data.
# Reinforcement Learning: In this type of learning, the algorithm learns by interacting with an environment and receiving feedback in the form of rewards or penalties.
# Applications of Machine Learning

# Image Recognition: Machine learning algorithms can be used to recognize objects, faces, and patterns in images.
# Natural Language Processing (NLP): Machine learning algorithms can be used to analyze and understand human language, including text and speech recognition.
# Predictive Analytics: Machine learning algorithms can be used to predict future outcomes based on historical data and trends.
# Speech Recognition: Machine learning algorithms can be used to recognize spoken words and convert them into text.
# Machine Learning Algorithms

# Decision Trees: A decision tree is a tree-like model that is trained on data, where each node represents a feature or attribute of the data.
# Random Forests: Random forests are an ensemble of multiple decision trees, which can be used to improve the accuracy and robustness of machine learning models.
# Support Vector Machines (SVMs): SVMs are a type of algorithm that can be used for classification and regression tasks.
# Challenges in Machine Learning

# Data Quality: The quality of the data used to train machine learning algorithms is crucial for their accuracy and reliability.
# Overfitting: Overfitting occurs when a machine learning model is too complex and fits the training data too closely, resulting in poor performance on new, unseen data.
# Bias and Fairness: Machine learning models can perpetuate biases present in the data, leading to unfair outcomes.
# Future of Machine Learning

# Explainability: Developing algorithms that can explain their decisions and provide insights into how they arrive at a conclusion.
# Adversarial Attacks: Developing defenses against adversarial attacks, which are designed to mislead machine learning models.
# Edge AI: Developing machine learning models that can run on edge devices, such as smartphones and smart home devices.
# Overall, machine learning has the potential to revolutionize various industries by enabling computers to learn from data and make predictions, classify objects, and provide recommendations. However, it also poses challenges in terms of data quality, overfitting, bias, and fairness, which must be addressed through further research and development.
