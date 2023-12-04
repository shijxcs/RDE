#pragma once

#include "fastXML.h"

class PfParam : public Param 
{
public:
	_bool pfswitch;
	_float gamma;
	_float alpha;

	PfParam(): Param()
	{
		pfswitch = false;
		gamma = 30;
		alpha = 0.8;
	}

	PfParam(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);
		fin >> (*this);
		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);
		fout << (*this);
		fout.close();
	}

	friend istream& operator>>( istream& fin, PfParam& pfparam )
	{
		fin >> static_cast<Param&>( pfparam );
		fin >> pfparam.pfswitch;
		fin >> pfparam.gamma;
		fin >> pfparam.alpha;
		return fin;
	}

	friend ostream& operator<<( ostream& fout, const PfParam& pfparam )
	{
		fout << static_cast<const Param&>( pfparam );
		fout << pfparam.pfswitch << "\n";
		fout << pfparam.gamma << "\n";
		fout << pfparam.alpha << endl;
		return fout;
	}
};

