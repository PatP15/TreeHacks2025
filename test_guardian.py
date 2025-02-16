from scrapybara import Scrapybara
from scrapybara.anthropic import Anthropic
from scrapybara.tools import BashTool, BrowserTool, ComputerTool, EditTool
from scrapybara.prompts import BROWSER_SYSTEM_PROMPT
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the Scrapybara client
client = Scrapybara(
    api_key=os.getenv("SCRAPYBARA_API_KEY"),
    timeout=6000,  # Set timeout to 600 seconds
)

# Start an Ubuntu instance
instance = client.start_ubuntu()

try:
    # Initialize tools
    tools = [
        BashTool(instance),
        ComputerTool(instance),
        EditTool(instance),
        BrowserTool(instance),
    ]

    # Initialize the Anthropic model
    model = Anthropic()

    # Define the repository URL and the script to run
    repo_url = "https://github.com/PatP15/TreeHacks2025.git"
    script_name = "app_demo.py"

    # Clone the GitHub repository
    output = instance.bash(command=f"git clone {repo_url}")
    print(f"Cloned repository: {repo_url}")
    print(output)

    # Extract the repository name from the URL
    repo_name = repo_url.split('/')[-1].replace('.git', '')

    # Navigate to the repository directory and run the Python script
    output = instance.bash(command=f"cd {repo_name} && pip install -r requirements.txt && python3 {script_name}")
    print(f"Executed script: {script_name} in {repo_name}")
    print(output)

    # Start the browser
    cdp_url = instance.browser.start().cdp_url
    print("Browser started.")

    from playwright.sync_api import sync_playwright
    # Navigate to the locally hosted web application
    playwright = sync_playwright().start()
    browser = playwright.chromium.connect_over_cdp(cdp_url)

    page = browser.new_page()
    page.goto("localhost:5001")


    # Define the prompt for the Anthropic model
    prompt = """
    You are an AI assistant tasked with interacting with the web application that monitors a persons health 
    through the dashboard. Your name is Capy Guardian (like a guardian angel). Your goal is to evaluate what is happening to the person and provide a write up 
    to the user. Here are the steps you should follow:

    Take multiple pictures of what you see on the dashboard and describe the different sections of the website.
    From teh camera, try to determine if the person is in a safe environment and if there are any signs of distress.
    From the heartbeat signal, try to determine if the person is in a healthy state or if there are any abnormalities.
    From the map streaming, the user is the green dot and the red dots are other people. Try to determine if the person is in a safe location.
    The real-time kin is a video feed of the person's joint angles. Try to determine if the person is moving normally or if there are any abnormalities.

    Here is a brief description of the different sections of the website:
    1. Dashboard View:
   - RealTime Kin: Shows real-time video/image feed
   - Chest Camera: Secondary camera view with caption capabilities
   - Map Streaming: Displays location data with what appears to be a facility map
   - Heartbeat Signal (PPG): Presumably for monitoring vital signs

    2. Logs Section:
    - Contains "Capy Guardian Logs"
    - Has a text input field for logging messages
    - Includes a "Send" button for submitting log entries

    Also you may need to make the website full screen to see all the sections.
    You will have to scroll around the website to view all the sections and gather the necessary information!

    Please proceed step by step, describing your actions and observations. Once you have gathered all the necessary information, provide a summary of the person's health status
    by submitting a write-up to the user through the "Send" button in the Logs section.
    Also determine if the person would require any immediate assistance or medical attention.
    If so, make it very clear in your write-up.

    Specificaaly look at the video if it seems it is persepctive from the ground then person may have fallen.
    The heartbeat signal is red if there is afib.

    Do not ask for any additional information from the user. Use only the information available on the website.

    Once you submit the logs leave the browser open.

    Open a text editor and write a summary of the person's health status based on the information you have gathered.    """

    # Use the Anthropic model to interact with the website
    response = client.act(
        model=model,
        tools=tools,
        system=BROWSER_SYSTEM_PROMPT,
        prompt=prompt,
        on_step=lambda step: print(f"\nStep output: {step.text}"),
    )

    # Output the final response from the model
    print("\nFinal response from the model:")
    print(response.output)


finally:
    print("Stopping the instance...")
    # Stop the instance to free up resources
    # instance.stop()
    # print("Instance stopped.")
