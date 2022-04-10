import sys


def progressBar(idx, total, bar_length=20):
    """
    progress bar
    ---
    Args
        idx: current client index or number of trained clients till now
        total: total number of clients
        phase: Train or Test
        bar_length: length of progress bar
    """
    percent = float(idx) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r> Train clients: [{}] {}% ({}/{}) ".format(
        arrow + spaces, int(round(percent * 100)), idx, total)
    )
    sys.stdout.flush()