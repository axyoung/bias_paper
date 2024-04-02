# bias_paper


This github repo is under construction - currently is being used to make code and data available for bias project.
Note for the OpenAI API scripts, for use to run them successfully they need a const.py in the appropriate folder with their api_key and the model names.
The pandas and openai libraries will need to be installed in your environment.

Contact authors for full original data output and cleaning scripts.


In order to run the current chatgpt_template files a python script like this should suffice.
```
import chatgpt_template
import const
import sys

if sys.argv[1] == "chatgpt_bias_ft_t5_aug":
    chatgpt_template.run(data='../data/test_data/test_data.csv', model=const.stereo_bias_aug_t5_model,
                         bias=True, norm=False, output_file='../data/ft_results/ft-on-aug_t5/ft_bias_base.csv')
```
For Bert and DistilBERT:
The scripts are located in the notebooks in the corresponding folder. The correct libraries will be setup by following the notebook
