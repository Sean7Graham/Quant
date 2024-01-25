# main.py

from src.data.data_fetcher import fetch_data

#
#
#


def main():
    try:
        data = fetch_data()

        # ...rest of workflow here

    except Exception as e:
        print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
