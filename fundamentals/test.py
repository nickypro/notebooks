""" Ensure good functioning """
import utils

def test():
    try:
        data = [1,2,3,4,5,6]
        return data

    except ValueError as ve:
        print(f"ValueError occurred: {ve}. This usually happens when a function receives an argument of the right type but inappropriate value.")
    except TypeError as te:
        print(f"TypeError occurred: {te}. This error is raised when an operation or function is applied to an object of inappropriate type.")
    except IndexError as ie:
        print(f"IndexError occurred: {ie}. This occurs when trying to access an index that is out of range for a list or tuple.")
    except KeyError as ke:
        print(f"KeyError occurred: {ke}. This happens when trying to access a dictionary with a key that does not exist.")
    except AttributeError as ae:
        print(f"AttributeError occurred: {ae}. This error is raised when an invalid attribute reference is made.")
    except ImportError as ie:
        print(f"ImportError occurred: {ie}. This indicates that a module could not be imported.")
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError occurred: {fnfe}. This error is raised when trying to access a file that does not exist.")
    except ZeroDivisionError as zde:
        print(f"ZeroDivisionError occurred: {zde}. This happens when attempting to divide by zero.")
    except OSError as ose:
        print(f"OSError occurred: {ose}. This error is raised when a system-related operation fails.")
    except RuntimeError as re:
        print(f"RuntimeError occurred: {re}. This error is raised when an error is detected that doesn't fall in any of the other categories.")
    except MemoryError as me:
        print(f"MemoryError occurred: {me}. This happens when an operation runs out of memory.")
    except StopIteration as si:
        print(f"StopIteration occurred: {si}. This indicates that an iterator has no further items to provide.")
    except AssertionError as ae:
        print(f"AssertionError occurred: {ae}. This error is raised when an assert statement fails.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Please check the code for any other issues.")
    finally:
        utils.close_context()

if __name__ == "__main__":
    print(test())
