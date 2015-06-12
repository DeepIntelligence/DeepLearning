#pragma once

#include <emmintrin.h>
#include <xmmintrin.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <limits>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

template<class T>
void Swap(T a, T b) {
	T t = a;
	a = b;
	b = t;
}

#define ASSERT(TST) ( (TST) ? (void)0 : (std::cerr << __FILE__ "(" << __LINE__	<< "): Assertion failed " #TST << std::endl,abort()) )

static const double INFTY = std::numeric_limits<double>::infinity();

static const double NaN = std::numeric_limits<double>::quiet_NaN();

static const double TOL = pow(std::numeric_limits<double>::epsilon(), (double)1.0 / 3);

static bool IsClose(double a, double b) {
	return abs(a - b) < TOL;
}

static bool IsNaN(double x) { return boost::math::isnan(x); }

static bool IsInf(double x) { return boost::math::isinf(x); }

static bool IsDangerous(double x) { return IsNaN(x) || IsInf(x); }

static double LogSum(double x, double y) {
	double d = x - y;
	if (d < -30) return y;
	else if (d > 30) return x;
	else if (d > 0) return x + log(1.0 + exp(-d));
	else return y + log(1.0 + exp(d));
}

static double Logistic(double x) {
	if (x < -30) return 0;
	else if (x > 30) return 1;
	else return 1.0 / (1.0 + exp(-x));
}

static double LogLoss(double x) {
	if (x < -30) return -x;
	else if (x > 30) return 0;
	else return log(1 + exp(-x));
}

template<class C>
void Serialize(const C & c, const string & filename) {
	ofstream outStream(filename, ios::out|ios::binary);
	if (!outStream.is_open()) {
		cout << "Couldn't open serialized file " << filename.c_str() << endl;
		exit(1);
	}

	c.Serialize(outStream);

	outStream.close();
}

template<class C>
void Deserialize(C & c, const string & filename) {
	ifstream inStream(filename, ios::in|ios::binary);
	if (!inStream.is_open()) {
		cout << "Couldn't open serialized file " << filename.c_str() << endl;
		exit(1);
	}

	c.Deserialize(inStream);

	inStream.close();
}
