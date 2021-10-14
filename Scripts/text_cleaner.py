#Author: Matt Williams
#Version: 10/14/2021


from bs4 import BeautifulSoup


def clean_html(html): 
    #TODO look into get_text parameters
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()




