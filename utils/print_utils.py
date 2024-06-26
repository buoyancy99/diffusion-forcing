from colorama import Fore


def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"
