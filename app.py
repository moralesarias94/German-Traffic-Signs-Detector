import click
import requests, zipfile, io, os

#Helper functions
def load_images(path):
    pass


@click.group()
def cli():
    pass


@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hello %s!' % name)

@cli.command()
@click.option('-m', default='logistic_regression_sklearn', help='Model to use e.g logistic_regression_sklearn')
@click.option('-d', default='/images/train/',
              help='Path to directory with trainig data')
def train(model, directory):
    pass

@cli.command()
@click.option('-m', default='logistic_regression_sklearn', help='Model to use e.g logistic_regression_sklearn')
@click.option('-d', default='/images/train/',
              help='Path to directory with testing data')
def test(model, directory):
    pass

@cli.command()
def download():
    """This function downloads the training and test images 
    from  from http://benchmark.ini.rub.de/Dataset_GTSDB/"""

    data_sets_urls = {'test': 'http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip',
                    'train': 'http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip'}
    base_output_dir = './images/'
    for data_set, url in data_sets_urls.items():
        click.echo(f'Sending request for {data_set} data')
        file_request = requests.get(url)
        if(file_request.ok):
            click.echo(f'Succees reading {data_set} data')
            zip_data_set = zipfile.ZipFile(io.BytesIO(file_request.content))
            output_dir = os.path.join(base_output_dir, data_set)
            for zip_info in zip_data_set.infolist():
                #Ignore directories
                if zip_info.filename[-1] == '/':
                    continue
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_data_set.extract(zip_info, output_dir)
        else:
            click.echo(f'Error reading {data_set} data')
    print('Finish')

if __name__ == '__main__':
    cli()
