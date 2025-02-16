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
    output = instance.bash(command=f"cd {repo_name} && python3 {script_name}")
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
    You are an AI assistant tasked with interacting with the web application hosted at http://localhost:8000.
    Your objectives are:
    1. Navigate through the website to understand its structure.
    2. Identify and extract key information presented on the site.
    3. Perform any available actions or form submissions to test functionality.
    4. Provide a summary of the site's features and any recommendations for improvement.

    Please proceed step by step, describing your actions and observations.
    """

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
    # Stop the instance to free up resources
    instance.stop()
    print("Instance stopped.")
