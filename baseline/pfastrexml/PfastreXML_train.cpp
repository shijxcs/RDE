#include <iostream>
#include <fstream>
#include <string>
#include <thread>

#include "timer.h"
#include "PfastreXML.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./PfastreXML_train [feature file name] [label file name] [inverse propensity file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 100 -g 30 -a 0.8 -q 1"<<endl<<endl;

	cerr<<"-S PfastXML switch, setting this to 1 omits tail classifiers, thus leading to PfastXML algorithm. default=0"<<endl;
	cerr<<"-T Number of threads to use. default=1"<<endl;
	cerr<<"-s Starting tree index. default=0"<<endl;
	cerr<<"-t Number of trees to be grown. default=50"<<endl;
	cerr<<"-b Feature bias value, extre feature value to be appended. default=1.0"<<endl;
	cerr<<"-c SVM weight co-efficient. default=1.0"<<endl;
	cerr<<"-m Maximum allowed instances in a leaf node. Larger nodes are attempted to be split, and on failure converted to leaves. default=10"<<endl;
	cerr<<"-l Number of label-probability pairs to retain in a leaf. default=100"<<endl;
	cerr<<"-g gamma parameter appearing in tail label classifiers. default=30"<<endl;
	cerr<<"-a Trade-off parameter between PfastXML and tail label classifiers. default=0.8"<<endl;
	cerr<<"-q quiet option (0/1). default=0"<<endl;

	cerr<<"feature and label files are in sparse matrix format"<<endl;
	exit(1);
}

PfParam parse_param(_int argc, char* argv[])
{
	PfParam pfparam;

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if(opt=="-m")
			pfparam.max_leaf = (_int)val;
		else if(opt=="-l")
			pfparam.lbl_per_leaf = (_int)val;
		else if(opt=="-b")
			pfparam.bias = (_float)val;
		else if(opt=="-c")
			pfparam.log_loss_coeff = (_float)val;
		else if(opt=="-T")
			pfparam.num_thread = (_int)val;
		else if(opt=="-s")
			pfparam.start_tree = (_int)val;
		else if(opt=="-t")
			pfparam.num_tree = (_int)val;
		else if(opt=="-S")
			pfparam.pfswitch = (_bool)val;
		else if(opt=="-g")
			pfparam.gamma = (_float)val;
		else if(opt=="-a")
			pfparam.alpha = (_float)val;
		else if(opt=="-q")
			pfparam.quiet = (_bool)val;
	}

	return pfparam;
}

int main(int argc, char* argv[])
{
	if(argc < 5)
		help();


	string ft_file = string(argv[1]);
	check_valid_filename(ft_file, true);
	SMatF* trn_X_Xf = new SMatF(ft_file);

	string lbl_file = string(argv[2]);
	check_valid_filename(lbl_file, true);
	SMatF* trn_X_Y = new SMatF(lbl_file);

	string prop_file = string(argv[3]);
	check_valid_filename(prop_file, true);
	ifstream fin;
	fin.open(prop_file);
	VecF inv_props;
	for(_int i=0; i<trn_X_Y->nr; i++)
	{
		_float f;
		fin>>f;
		inv_props.push_back(f);
	}
	fin.close();

	string model_folder = string(argv[4]);
	check_valid_foldername(model_folder);

	PfParam pfparam = parse_param(argc-5,argv+5);
	pfparam.num_Xf = trn_X_Xf->nr;
	pfparam.num_Y = trn_X_Y->nr;
	pfparam.write(model_folder+"/param");

	if( pfparam.quiet )
		loglvl = LOGLVL::QUIET;

	USE_IDCG = false;

	_float train_time = 0;
	Timer timer;

	timer.tic();
	/* Weighting label matrix with inverse propensity weights */
	for(_int i=0; i<trn_X_Y->nc; i++)
		for(_int j=0; j<trn_X_Y->size[i]; j++)
			trn_X_Y->data[i][j].second *= inv_props[trn_X_Y->data[i][j].first];
	train_time += timer.toc();

	/* training PfastXML trees */
	_float tmptime;
	train_trees(trn_X_Xf, trn_X_Y, pfparam, model_folder, tmptime);
	train_time += tmptime;

	/* if pfswitch is true, terminate here immediately after PfastXML */
	if(pfparam.pfswitch)
	{
		cout << "training time: " << train_time/3600.0 << " hr" << endl;

		delete trn_X_Xf;
		delete trn_X_Y;
		return 0;
	}

	timer.tic();

	/* normalize feature vectors to unit norm */
	trn_X_Xf->unit_normalize_columns();

	/*--- calculating model parameters saved in w ---*/

	SMatF* tmat = trn_X_Y->transpose();

	for(int i=0; i<tmat->nc; i++)
	{
		_float a = 1.0/(tmat->size[i]);
		for(int j=0; j<tmat->size[i]; j++)
			tmat->data[i][j].second = a;
	}

	SMatF* w = trn_X_Xf->prod(tmat);

	train_time += timer.toc();

	cout << "training time: " << train_time/3600.0 << " hr" << endl;


	w->write(model_folder+"/w");

	/* free allocated resources */
	delete tmat;
	delete w;
	delete trn_X_Xf;
	delete trn_X_Y;
}
