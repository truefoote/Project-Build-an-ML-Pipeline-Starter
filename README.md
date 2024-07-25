# Build an ML Pipeline for Short-Term Rental Prices in NYC
You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

## Links
Weights & Biases:
https://wandb.ai/truefoote-western-governors-university/nyc_airbnb?nw=nwusertruefoote

Github:
https://github.com/truefoote/Project-Build-an-ML-Pipeline-Starter


## Table of contents

- [Preliminary steps](#preliminary-steps)
  * [Fork the Starter Kit](#fork-the-starter-kit)
  * [Create environment](#create-environment)
  * [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  * [The configuration](#the-configuration)
  * [Running the entire pipeline or just a selection of steps](#Running-the-entire-pipeline-or-just-a-selection-of-steps)
  * [Pre-existing components](#pre-existing-components)
- [Steps](#steps)
  *[EDA](#step-1-exploratory-data-analysis)
  *[Data Cleaning](#step-2-data-cleaning)
  *[Data Testing](#step-3-data-testing)
  *[Initial Training](#step-4-initial-training)
  *[Model Selection and Test](#step-5-model-selection-and-test)
  *[Pipeline Release and Updates](#step-6-pipeline-release-and-updates)

## Preliminary steps
### Fork the Starter kit
Go to [https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter)
and click on `Fork` in the upper right corner. This will create a fork in your Github account, i.e., a copy of the
repository that is under your control. Now clone the repository locally so you can start working on it:

```
git clone https://github.com/[your github username]/Project-Build-an-ML-Pipeline-Starter.git
```

and go into the repository:

```
cd Project-Build-an-ML-Pipeline-Starter
```
Commit and push to the repository often while you make progress towards the solution. Remember 
to add meaningful commit messages.

### Create environment
Make sure to have conda installed and ready, then create a new environment using the ``environment.yaml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

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


### The configuration
As usual, the parameters controlling the pipeline are defined in the ``config.yaml`` file defined in
the root of the starter kit. We will use Hydra to manage this configuration file. 
Open this file and get familiar with its content. Remember: this file is only read by the ``main.py`` script 
(i.e., the pipeline) and its content is
available with the ``go`` function in ``main.py`` as the ``config`` dictionary. For example,
the name of the project is contained in the ``project_name`` key under the ``main`` section in
the configuration file. It can be accessed from the ``go`` function as 
``config["main"]["project_name"]``.

NOTE: do NOT hardcode any parameter when writing the pipeline. All the parameters should be 
accessed from the configuration file.

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

### Pre-existing components
In order to simulate a real-world situation, we are providing you with some pre-implemented
re-usable components. While you have a copy in your fork, you will be using them from the original
repository by accessing them through their GitHub link, like:

```python
_ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
```
where `config['main']['components_repository']` is set to 
[https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/tree/main/components](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/tree/main/components).
You can see the parameters that they require by looking into their `MLproject` file:

- `get_data`: downloads the data. [MLproject](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/blob/main/components/get_data/MLproject)
- `train_val_test_split`: segrgate the data (splits the data) [MLproject](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter/blob/main/components/train_val_test_split/MLproject)

## In case of errors
When you make an error writing your `conda.yml` file, you might end up with an environment for the pipeline or one
of the components that is corrupted. Most of the time `mlflow` realizes that and creates a new one every time you try
to fix the problem. However, sometimes this does not happen, especially if the problem was in the `pip` dependencies.
In that case, you might want to clean up all conda environments created by `mlflow` and try again. In order to do so,
you can get a list of the environments you are about to remove by executing:

```
> conda info --envs | grep mlflow | cut -f1 -d" "
```

If you are ok with that list, execute this command to clean them up:

**_NOTE_**: this will remove *ALL* the environments with a name starting with `mlflow`. Use at your own risk

```
> for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

This will iterate over all the environments created by `mlflow` and remove them.

# Steps
## Step 1: Exploratory Data Analysis
The scope of this section is to get an idea of how the process of an EDA works in the context of pipelines during the data exploration phase. In a real scenario, you would spend a lot more time in this phase, but here we are going to do the bare minimum.

### Download data
The main.py script already comes with the download step implemented. Run the pipeline to get a sample of the data. The pipeline will also upload it to Weights & Biases:

```
> mlflow run . -P steps=download
```

You will see a message similar to:

```
2021-03-12 15:44:39,840 Uploading sample.csv to Weights & Biases
```

This tells you that the data is going to be stored in W&B as the artifact named ``sample.csv``.

### EDA
Go to the ``src/eda`` folder in the starter kit. Open the Jupyter Notebook file ``eda.ipynb``. The EDA process has been implemented for you. You don't need to modify the code. Follow the instructions in the notebook and run the cells to complete the EDA step.

To work with Jupyter Notebooks, you need first to run the following command in the terminal to spin up a Jupyter Lab instance.

```
> jupyter-lab
```

Once you save the notebook, close the notebook by clicking File -> Close and Shutdown Notebook. Once the notebook is shut down, click File -> Shutdown to stop the Jupyter Lab instance.

## Step 2: Data Cleaning
Now we transfer what we have done in EDA to a new "basic cleaning" step that cleans the ``sample.csv`` artifact and creates a new ``clean_sample.csv`` with the cleaned data.

Go to ``src/basic_cleaning``, containing the files required for an MLflow step:

* ``conda.yml``: conda environment for the step
* ``MLproject``: parameters and definitions of the step
* ``run.py``: script of the step

### Understand arguments in MLproject
Check the arguments stored in the ``src/basic_cleaning/MLproject``. These are the arguments passed to the ``basic_cleaning`` step.

* ``input_artifact``: the input artifact
* ``output_artifact``: the name for the output artifact
* ``output_type``: the type for the output artifact
* ``output_description``: a description of the output artifact
* ``min_price``: the minimum price to consider
* ``max_price``: the maximum price to consider

All parameters should be of type ``str`` except ``min_price`` and ``max_price`` which should be ``float``.

### Add arguments information in run.py
Using the argument information you found above, fill in the missing type and description of the arguments in ``src/basic_cleaning/run.py``.

Note the comments like ``TODO``, ``INSERT TYPE HERE`` and ``INSERT DESCRIPTION HERE``. Do not change code with comments like ``DO NOT MODIFY``.

### Add the step to the pipeline
Note the following comment in ``main.py``. That is where you implement the code:

```python
##################
# Implement here #
##################
```

Add the code below to the ``basic_cleaning step`` in the pipeline (the ``main.py`` file). You don't need to modify the code.

```python
if "basic_cleaning" in active_steps:
    _ = mlflow.run(
         os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
         "main",
         parameters={
             "input_artifact": "sample.csv:latest",
             "output_artifact": "clean_sample.csv",
             "output_type": "clean_sample",
             "output_description": "Data with outliers and null values removed",
             "min_price": config['etl']['min_price'],
             "max_price": config['etl']['max_price']
         },
     )
```

Please note how the path to the step is constructed:

```python
os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning")
```

This is necessary because Hydra executes the script in a different directory than the root of the starter kit. **You will have to do the same for every step that you will add to the pipeline.**

Remember that when you refer to an artifact stored on W&B, you **MUST** specify a version or a tag. For example, here, the ``input_artifact`` should be ``sample.csv:latest`` and **NOT** just ``sample.csv``.

If you forget to do this, you will see a message like
```
Attempted to fetch artifact without alias (e.g. "<artifact_name>:v3" or "<artifact_name>:latest")
```

### Run the step
Run the pipeline with the following command:

```
> mlflow run . -P steps=basic_cleaning
```

If you go to W&B, you will see the new artifact type ``clean_sample`` and within it the ``clean_sample.csv`` artifact.

## Step 3: Data Testing
After the cleaning, it is a good practice to put some tests that verify that the data does not contain surprises. In this section, you will work in ''src/data_check'' to write a few tests and add it to the ML pipeline.

### Create a reference dataset
One of our tests will compare the distribution of the current data sample with a reference, to ensure that there is no unexpected change. Therefore, we first need to define a "reference dataset". We will just tag the latest ''clean_sample.csv'' artifact on W&B as our reference dataset.

1. Go to https://wandb.ai/home in your web browser.
2. Navigate to your project, then to the artifact tab.
3. Click on "clean_sample", then on the version with the ``latest tag``. This is the last one we produced in the previous basic cleaning step.
4. Add a tag reference to it by clicking the "+" in the Aliases section on the right.
5. Add a tag ``reference`` by clicking on the "+" button next to "Aliases" for the artifact overview.

### Create tests
Now we are ready to add some tests. In the starter kit, you can find a ``src/data_check`` step that you need to complete two tests.

* ``test_row_count``
* ``test_price_range``

Note the following comment in the ``src/data_check/test_data.py``. That is where you add the tests.

```python
########################################################
# Implement here test_row_count and test_price_range   #
########################################################
```

### test_row_count
Let's start by appending to ``test_data.py`` the following test:

```python
def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000
which checks that the size of the dataset is reasonable (not too small, not too large).
```

### test_price_range
Now, add another test ``test_price_range(data, min_price, max_price)`` that checks that the price range is between ``min_price`` and ``max_price``.

**Hint**: you can use the ``data['price'].between(...)`` method.

Also, remember that we are using closures, so the name of the variables that your test takes in MUST BE exactly ``data``, ``min_price`` and ``max_price``.

### Add the step to the pipeline
Now add the ``data_check`` step to the ``main.py`` file, so that it gets executed as part of our pipeline.

**Hint**:

Check the ``basic_cleaning`` step you implemented before as an example. The implementation is very similar.
Check the ``src/data_check/MLproject`` for required arguments
Use ``clean_sample.csv:latest`` as ``csv`` and ``clean_sample.csv:reference`` as ``ref``. Right now, they point to the same file, but later on, they will not: we will fetch another sample of data and therefore the ``latest`` tag will point to that.

Also, use the value stored in ``config.yaml`` for the other parameters. For example, use ``config["data_check"]["kl_threshold"]`` for the ``kl_threshold`` parameter.

### Run the step
Run the pipeline and ensure the tests are executed and passed. Remember that you can run just this step with:

```
> mlflow run . -P steps=data_check
```

You can safely ignore the following DeprecationWarning if you see it:

```
DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' 
is deprecated since Python 3.3, and in 3.10 it will stop working
```

If your data pass all the tests, you should see a message similar to

```
test_data.py::test_column_names PASSED                   [ 16%]
test_data.py::test_neighborhood_names PASSED             [ 33%]
test_data.py::test_proper_boundaries PASSED              [ 50%]
test_data.py::test_similar_neigh_distrib PASSED          [ 66%]
test_data.py::test_row_count PASSED                      [ 83%]
test_data.py::test_price_range PASSED                    [100%]
```

## Step 4: Initial Training
Now the data is cleaned and checked. It's time to train the model. In this section, you will work in the following files and folders:

* ``components/train_val_test_split``: splitting data into training and test dataset
* ``src/train_random_forest``: construct pipelines to train a model and make inferences on the model

You will complete the code for these two steps and add them to the ML pipeline.

### Data splitting
The step to split the training and testing dataset has been provided to you in ``components/train_val_test_split``.

### Add the step to the pipeline
Add this step to the ``main.py`` under the ``data_split`` step. You can see the parameters accepted by this step in ``components/train_val_test_split/MLproject``.

Since this step is a component in the ``components`` folder, the path to the step can be expressed as:

```python
_ = mlflow.run(
    f"{config['main']['components_repository']}/train_val_test_split",
    'main',
    parameters = {
             ...
    }
)
```

As usual, for parameters like ``test_size``, ``random_seed`` and ``stratify_by``, look at the ``modeling`` section in the ``config.yaml`` file. For ``input``, you can use ``clean_sample.csv:latest``.

**Hint**: The implementation of ``data_split`` is very similar to the ``download`` step.

### Run the step
Now run the step.

After you execute, you will see something like:

```
2021-03-15 01:36:44,818 Uploading trainval_data.csv dataset
2021-03-15 01:36:47,958 Uploading test_data.csv dataset
```

This tells you that the script is uploading two new datasets: ``trainval_data.csv`` and ``test_data.csv``.

If you go to W&B, you will see a new artifact type ``TEST_DATA`` and within it the ``test_data.csv`` artifact. And an artifact type ``TRAINVAL_DATA`` and within it the ``trainval_data.csv`` artifact.

### Train Random Forest
#### Complete ``run.py``
Read the script ``src/train_random_forest/run.py`` carefully and complete the following missing pieces.

* Build a preprocessing pipeline that imputes missing values and encodes the variable
* Build the inference pipeline called "sk_pipe"
* Fit the "sk_pipe" pipeline
* Save the "sk_pipe" pipeline
* Save model metrics MAE and R2

All the places where you need to insert code are marked by a ``# YOUR CODE HERE`` comment and are delimited by two signs like ``######################################``. For example:

```python
######################################
# Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
# YOUR CODE HERE
######################################
```

You can find further instructions in the ``run.py`` file.

#### Add the step to the pipeline
Once you are done, add this step to ``main.py`` under the ``train_random_forest`` step. As usual, you can check ``src/train_random_forest/MLproject`` for all the required arguments.

**Hints**:

1. The implementation should be similar to ``basic_cleaning`` and ``data_check`` steps.
2. Use ``trainval_data.csv:latest`` as ``trainval_artifact``.
3. Use the ``name random_forest_export`` as ``output_artifact``.
4. The ``main.py`` already provides a ``variable rf_config`` to be passed as the ``rf_config`` parameter.
5. Check the ``modeling`` section in ``config.yaml`` for the other parameters.

#### Run the step with hyperparameters optimization
Use the code below to run the training step with varying the hyperparameters of the Random Forest model. **Note**: this step may take a while to complete.

```
> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
```

This is done by exploiting the Hydra configuration system. It uses the multi-run feature (adding the ``-m`` option at the end of the ``hydra_options`` specification), and sets the parameter ``modeling.random_forest.max_depth`` to 10, 50, and the ``modeling.random_forest.n_estimators`` to 100, 200.

## Step 5: Model Selection and Test
You've trained the model with different hyperparameters. Now you can select the best model and test the model with the test dataset. In this section, you will work in W&B to choose the best model and test the model by adding the test step ``components/test_regression_model`` to the ML pipeline.

### Select the best model
Go to W&B and select the best performing model.

Look for your best model within W&B

**HINT**: you should switch to the Table view (second icon on the left), then click on the upper right on "columns", remove all selected columns by clicking on "Hide all", then click on the left list on "ID", "Job Type", "max_depth", "n_estimators", "mae" and "r2". Click on "Close". Now in the table view you can click on the "mae" column on the three little dots, then select "Sort asc". This will sort the runs by ascending Mean Absolute Error (best result at the top).

When you have found the best job, click on its name, then go to its artifacts and select the "model_export" output artifact. You can now add a ``prod`` tag to it to mark it as "production ready".

### Test the model
Use the provided step ``components/test_regression_model`` to test your production model against the test set.

### Add the step to the pipeline
Add this step in the ``main.py`` under ``test_regression_model`` step. As usual, you can see the parameters in the ``components/test_regression_model/MLproject`` file.

Use the artifact ``random_forest_export:prod`` for the parameter ``mlflow_model`` and the test artifact ``test_data.csv:latest`` as ``test_artifact``.

Hint: the implementation of this step is similar to the ``data_split`` step.

### Run the step
This step is NOT run by default when you run the pipeline. In fact, it needs the manual step of promoting a model to ``prod`` before it can complete successfully. That is what you have just done earlier. Now use the command line below to run the model:

```
> mlflow run . -P steps=test_regression_model
```

## Step 6: Pipeline Release and Updates
Now is the time to release the pipeline. In this section, you will first visualize the pipeline in W&B and then release the pipeline in your Github Repo. Next, you will use the released pipeline to train the model on a new dataset. Let's dive in.

### Visualize the pipeline
You can now go to W&B, and go to the Artifacts section. Select the model export artifact then click on the Lineage tab.

### Release the pipeline
First, copy the best hyperparameters you found into the ``config.yaml`` so they become the default values. Then, push your changes to the forked project repository on your Github repo. Finally, go to the repository on GitHub and make a release.

Call the release ``1.0.0``:

If you find problems in the release, fix them and then make a new release like ``1.0.1``, ``1.0.2`` and so on.

Train the model on a new data sample
Let's now test that we can run the release using ``mlflow`` without any other pre-requisite. We will train the model on a new sample of data that our company received (``sample2.csv``):

(be ready for a surprise, keep reading even if the command fails)

```python
> mlflow run https://github.com/[your github username]/Project-Build-an-ML-Pipeline-Starter.git \
             -v [the version you want to use, like 1.0.0] \
             -P hydra_options="etl.sample='sample2.csv'"
```

But, wait! It failed! The test ``test_proper_boundaries`` failed. Apparently, there is one point that is outside of the boundaries. This is an example of a "successful failure", i.e., a test that did its job and caught an unexpected event in the pipeline (in this case, in the data).

You can fix this by adding these two lines into the ``src/basic_cleaning/run.py`` just before the ``# Save the cleaned data`` section. You should see a ``TODO`` indication.

```python
idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()
```

This will drop rows in the dataset that are not in the proper geolocation.

Then commit your change, make a new release (for example ``1.0.1``), and rerun the pipeline (of course, you need to use ``-v 1.0.1`` when calling mlflow this time). Now the run should succeed, and you have trained your new model on the new data.

## License

[License](LICENSE.txt)
