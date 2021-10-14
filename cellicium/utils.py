def display(x):
    try:
        from IPython.display import display
        display(x)
    except ModuleNotFoundError:
        print(x)
