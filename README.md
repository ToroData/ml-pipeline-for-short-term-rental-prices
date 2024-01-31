# Build an ML Pipeline for Short-Term Rental Prices in NYC

You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

## Link to GitHub repo
[GitHub Repo](https://github.com/ToroData/ml-pipeline-for-short-term-rental-prices)


### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

### Running the entire pipeline or just a selection of steps
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
>  mlflow run .
```
This will run the entire pipeline.

When developing it is useful to be able to run one step at the time. Say you want to run only
the ``download`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```
If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

## Instructions

## Exploratory Data Analysis (EDA)
The scope of this section is to get an idea of how the process of an EDA works in the context of pipelines, during the data exploration phase. [HTML EDA Report](src/eda/report/eda-report.html). Then I transfer the data processing I have done as part of the EDA to a new `basic_cleaning` step that starts from the `sample.csv` artifact and create a new artifact `clean_sample.csv` with the cleaned data. 

## Data Testing
[Prod tag](/images/prod_tag.png)
After the cleaning, it is a good practice to put some tests that verify that the data does not contain surprises. Tested with `clean_sample.csv:latest`.
[Prod tag](/images/tests.png)

## Data Splitting
Use the provided component called `train_val_test_split` to extract and segregate the test set. Add it to the pipeline then run the pipeline. As usual, use the configuration for the parameters like `test_size`, `random_seed` and `stratify_by`.

## Training Random Forest and Optimize Hyperparameters

```
mlflow run .   -P steps=train_random_forest   -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"`
```
With that command I perform multiple experiments on a random forest.

## Select the Best Model
[Best model](/images/training.png)

## Test
[Test](/images/test.png)

## Visualize the Pipeline
[Lineage](/images/lineage.png)

## Release the Pipeline
I realease on Github realeases.

## Train the Model on a New Data Sample

To train a new model directly from released project, run:

```
mlflow run https://github.com/ToroData/ml-pipeline-for-short-term-rental-prices.git -v 1.0.0 -P hydra_options="etl.sample='sample2.csv'"
```
[Test failed](/images/test_failed.png)

Test failed due to coordinate boundaries

## Release the fixed project
Once corrected in the cleanup step, he launched again. Subsequently, I evaluated it with the following command and all the tests passed.

```
mlflow run https://github.com/ToroData/ml-pipeline-for-short-term-rental-prices.git -v 1.1.0 -P hydra_options="etl.sample='sample2.csv'"
```

[Test](/images/tests.png)

## Hardware Report
[Hardware report](/Report%20of%20the%20hardware%20utilization%20_%20nyc_airbnb%20–%20Weights%20&%20Biases.pdf)
[Test](/images/hardware.png)

## License

[License](LICENSE.txt)
