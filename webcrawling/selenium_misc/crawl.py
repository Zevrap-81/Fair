from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

"""
Get the website html using Selenium.

Args:
    URL (str): The URL of the website to load.
    wait (bool, optional): Whether to wait for the page to load. Defaults to False.
    wait_selector (str, optional): The selector to wait for. Defaults to ''.

Returns:
    html (str): The HTML content of the website.
"""
def get_website(URL: str, wait=False, wait_selector='', cloudflare=False):

    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    if not cloudflare:
        options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-popup-blocking")

    # disable logging
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    

    driver = webdriver.Chrome(options=options)

    # use selenium_stealth to prevent cloudflare 
    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )

    # if clouidflare:
    #     driver.execute_script(f"window.open('{URL}', '_blank')")
    #     sleep(15)
    #     driver.switch_to.window(driver.window_handles[1])
    # else:
    # load website
    driver.get(URL)


    if wait:
        element_present_3 = EC.presence_of_element_located((By.ID, wait_selector))
        WebDriverWait(driver, 10000).until(element_present_3)
    

    html = driver.page_source

    driver.quit()

    # selenium.webdriver.remote.webelement.WebElement
    return html