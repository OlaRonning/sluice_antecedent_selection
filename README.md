# Sluice Antecedent Selection
Code for the NAACL HLT 2018 paper: Sluice resolution without hand-crafted features over brittle syntax tree. 

To train and evaluate:

	 > python3 mains/<model>_train_eval.py

model: rhs, sg or base. F.ex:

	 > python3 mains/rhs_train_eval.py

To evaluate on dialogue:

	 > python3 mains/eval_dialogue -l <log> -m <model>
*\<model\> must be trained before it can be evaluated on dialogue.* F.ex:

	 > python3 mains/eval_dialogue -l tensorboard/20180415_230245 -m base

# Dependencies
Run the command below to install depedencies.

	 > pip install -r requirements.txt

The dependencies are:
  * absl-py==0.1.13
  * astor==0.6.2
  * bleach==1.5.0
  * gast==0.2.0
  * grpcio==1.10.1
  * html5lib==0.9999999
  * Markdown==2.6.11
  * numpy==1.14.2
  * protobuf==3.5.2.post1
  * six==1.11.0
  * tensorboard==1.7.0
  * tensorflow==1.7.0 *# Note requirement.txt is not gpu supported version*
  * termcolor==1.1.0
  * Werkzeug==0.14.1


# Structure
The structure of the repo is:

```├── bin
├── configs
├── data
│   ├── auxiliaries
│   ├── esc
│   └── opensub
├── load
│   ├── core
│   └── readers
├── mains
├── metrics
└── nns
    └── custom_tf
```
