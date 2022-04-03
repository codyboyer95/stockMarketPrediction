from wwo_hist import retrieve_hist_data


def run():
    frequency=24
    start_date = '01-JAN-2009'
    end_date = '31-DEC-2021'
    api_key = '7b35bbbbcac3470b8d8213354220204'
    location_list = ['new_york_city']
    
    hist_weather_data = retrieve_hist_data(api_key,
                                    location_list,
                                    start_date,
                                    end_date,
                                    frequency,
                                    location_label = False,
                                    export_csv = True,
                                    store_df = True)

if __name__ == "__main__":
    run()
