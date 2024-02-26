#include "Common.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureProc.hpp"
#include "CommonAuxPixFormat.hpp"
#include "FastAriphmetics.hpp"

constexpr float COEFF(const float& val) noexcept { return val / 255.f; }

/* 
	Let's try ti use table with predefine RGB value for each CCT.
	In future this table should be replaced by Ohno(2013) or Robertson(2022) computational algorithm.
*/
static constexpr float cctCoeff[][3] =  /* 1000 ... 13000 Kelvins [step - 100 degrees], 2 degrees */
{
	{ COEFF(255.f), COEFF( 51.f), COEFF(  0.f) }, /* 1000  */
	{ COEFF(255.f), COEFF( 69.f), COEFF(  0.f) }, /* 1100  */
	{ COEFF(255.f), COEFF( 82.f), COEFF(  0.f) }, /* 1200  */ 
	{ COEFF(255.f), COEFF( 93.f), COEFF(  0.f) }, /* 1300  */
	{ COEFF(255.f), COEFF(102.f), COEFF(  0.f) }, /* 1400  */
	{ COEFF(255.f), COEFF(111.f), COEFF(  0.f) }, /* 1500  */
	{ COEFF(255.f), COEFF(118.f), COEFF(  0.f) }, /* 1600  */
	{ COEFF(255.f), COEFF(124.f), COEFF(  0.f) }, /* 1700  */
	{ COEFF(255.f), COEFF(130.f), COEFF(  0.f) }, /* 1800  */
	{ COEFF(255.F), COEFF(135.f), COEFF(  0.f) }, /* 1900  */
	{ COEFF(255.f), COEFF(141.f), COEFF( 11.f) }, /* 2000  */
	{ COEFF(255.f), COEFF(146.f), COEFF( 29.f) }, /* 2100  */
	{ COEFF(255.f), COEFF(152.f), COEFF( 41.f) }, /* 2200  */
	{ COEFF(255.f), COEFF(157.f), COEFF( 51.f) }, /* 2300  */
	{ COEFF(255.f), COEFF(162.f), COEFF( 60.f) }, /* 2400  */
	{ COEFF(255.f), COEFF(166.f), COEFF( 69.f) }, /* 2500  */
	{ COEFF(255.f), COEFF(170.f), COEFF( 77.f) }, /* 2600  */
	{ COEFF(255.f), COEFF(174.f), COEFF( 84.f) }, /* 2700  */
	{ COEFF(255.f), COEFF(178.f), COEFF( 91.f) }, /* 2800  */
	{ COEFF(255.f), COEFF(182.f), COEFF( 98.f) }, /* 2900  */
	{ COEFF(255.f), COEFF(185.f), COEFF(105.f) }, /* 3000  */
	{ COEFF(255.f), COEFF(189.f), COEFF(111.f) }, /* 3100  */
	{ COEFF(255.f), COEFF(192.f), COEFF(118.f) }, /* 3200  */
	{ COEFF(255.f), COEFF(195.f), COEFF(124.f) }, /* 3300  */
	{ COEFF(255.f), COEFF(198.f), COEFF(130.f) }, /* 3400  */
	{ COEFF(255.f), COEFF(201.f), COEFF(135.f) }, /* 3500  */
	{ COEFF(255.f), COEFF(203.f), COEFF(141.f) }, /* 3600  */
	{ COEFF(255.f), COEFF(206.f), COEFF(146.f) }, /* 3700  */
	{ COEFF(255.f), COEFF(208.f), COEFF(151.f) }, /* 3800  */
	{ COEFF(255.f), COEFF(211.f), COEFF(156.f) }, /* 3900  */
	{ COEFF(255.f), COEFF(213.f), COEFF(161.f) }, /* 4000  */
	{ COEFF(255.f), COEFF(215.f), COEFF(166.f) }, /* 4100  */
	{ COEFF(255.f), COEFF(217.f), COEFF(171.f) }, /* 4200  */
	{ COEFF(255.f), COEFF(219.f), COEFF(175.f) }, /* 4300  */
	{ COEFF(255.f), COEFF(221.f), COEFF(180.f) }, /* 4400  */
	{ COEFF(255.f), COEFF(223.f), COEFF(184.f) }, /* 4500  */
	{ COEFF(255.f), COEFF(225.f), COEFF(188.f) }, /* 4600  */
	{ COEFF(255.f), COEFF(226.f), COEFF(192.f) }, /* 4700  */
	{ COEFF(255.f), COEFF(228.f), COEFF(196.f) }, /* 4800  */
	{ COEFF(255.f), COEFF(229.f), COEFF(200.f) }, /* 4900  */
	{ COEFF(255.f), COEFF(231.f), COEFF(204.f) }, /* 5000  */
	{ COEFF(255.f), COEFF(232.f), COEFF(208.f) }, /* 5100  */
	{ COEFF(255.f), COEFF(234.f), COEFF(211.f) }, /* 5200  */
	{ COEFF(255.f), COEFF(235.f), COEFF(215.f) }, /* 5300  */
	{ COEFF(255.f), COEFF(237.f), COEFF(218.f) }, /* 5400  */
	{ COEFF(255.f), COEFF(238.f), COEFF(222.f) }, /* 5500  */
	{ COEFF(255.f), COEFF(239.f), COEFF(225.f) }, /* 5600  */
	{ COEFF(255.f), COEFF(240.f), COEFF(228.f) }, /* 5700  */
	{ COEFF(255.f), COEFF(241.f), COEFF(231.f) }, /* 5800  */
	{ COEFF(255.f), COEFF(243.f), COEFF(234.f) }, /* 5900  */
	{ COEFF(255.f), COEFF(244.f), COEFF(237.f) }, /* 6000  */
	{ COEFF(255.f), COEFF(245.f), COEFF(240.f) }, /* 6100  */
	{ COEFF(255.f), COEFF(246.f), COEFF(243.f) }, /* 6200  */
	{ COEFF(255.f), COEFF(247.f), COEFF(245.f) }, /* 6300  */
	{ COEFF(255.f), COEFF(248.f), COEFF(248.f) }, /* 6400  */
	{ COEFF(255.f), COEFF(255.f), COEFF(255.f) }, /* 6500  */
	{ COEFF(255.f), COEFF(249.f), COEFF(253.f) }, /* 6600  */
	{ COEFF(254.f), COEFF(250.f), COEFF(255.f) }, /* 6700  */
	{ COEFF(252.f), COEFF(248.f), COEFF(255.f) }, /* 6800  */
	{ COEFF(250.f), COEFF(247.f), COEFF(255.f) }, /* 6900  */
	{ COEFF(247.f), COEFF(245.f), COEFF(255.f) }, /* 7000  */
	{ COEFF(245.f), COEFF(244.f), COEFF(255.f) }, /* 7100  */
	{ COEFF(243.f), COEFF(243.f), COEFF(255.f) }, /* 7200  */
	{ COEFF(241.f), COEFF(241.f), COEFF(255.f) }, /* 7300  */
	{ COEFF(239.f), COEFF(240.f), COEFF(255.f) }, /* 7400  */
	{ COEFF(238.f), COEFF(239.f), COEFF(255.f) }, /* 7500  */
	{ COEFF(236.f), COEFF(238.f), COEFF(255.f) }, /* 7600  */
	{ COEFF(234.f), COEFF(237.f), COEFF(255.f) }, /* 7700  */
	{ COEFF(233.f), COEFF(236.f), COEFF(255.f) }, /* 7800  */
	{ COEFF(231.f), COEFF(234.f), COEFF(255.f) }, /* 7900  */
	{ COEFF(229.f), COEFF(233.f), COEFF(255.f) }, /* 8000  */
	{ COEFF(228.f), COEFF(233.f), COEFF(255.f) }, /* 8100  */
	{ COEFF(227.f), COEFF(232.f), COEFF(255.f) }, /* 8200  */
	{ COEFF(225.f), COEFF(231.f), COEFF(255.f) }, /* 8300  */
	{ COEFF(224.f), COEFF(230.f), COEFF(255.f) }, /* 8400  */
	{ COEFF(223.f), COEFF(229.f), COEFF(255.f) }, /* 8500  */
	{ COEFF(221.f), COEFF(228.f), COEFF(255.f) }, /* 8600  */
	{ COEFF(220.f), COEFF(227.f), COEFF(255.f) }, /* 8700  */
	{ COEFF(219.f), COEFF(226.f), COEFF(255.f) }, /* 8800  */
	{ COEFF(218.f), COEFF(226.f), COEFF(255.f) }, /* 8900  */
	{ COEFF(217.f), COEFF(225.f), COEFF(255.f) }, /* 9000  */
	{ COEFF(216.f), COEFF(224.f), COEFF(255.f) }, /* 9100  */
	{ COEFF(215.f), COEFF(223.f), COEFF(255.f) }, /* 9200  */
	{ COEFF(214.f), COEFF(223.f), COEFF(255.f) }, /* 9300  */
	{ COEFF(213.f), COEFF(222.f), COEFF(255.f) }, /* 9400  */
	{ COEFF(212.f), COEFF(221.f), COEFF(255.f) }, /* 9500  */
	{ COEFF(211.f), COEFF(221.f), COEFF(255.f) }, /* 9600  */
	{ COEFF(210.f), COEFF(220.f), COEFF(255.f) }, /* 9700  */
	{ COEFF(209.f), COEFF(220.f), COEFF(255.f) }, /* 9800  */
	{ COEFF(208.f), COEFF(219.f), COEFF(255.f) }, /* 9000  */
	{ COEFF(207.f), COEFF(218.f), COEFF(255.f) }, /* 10000 */
	{ COEFF(207.f), COEFF(218.f), COEFF(255.f) }, /* 10100 */
	{ COEFF(206.f), COEFF(217.f), COEFF(255.f) }, /* 10200 */
	{ COEFF(205.f), COEFF(217.f), COEFF(255.f) }, /* 10300 */
	{ COEFF(204.f), COEFF(216.f), COEFF(255.f) }, /* 10400 */
	{ COEFF(204.f), COEFF(216.f), COEFF(255.f) }, /* 10500 */
	{ COEFF(203.f), COEFF(215.f), COEFF(255.f) }, /* 10600 */
	{ COEFF(202.f), COEFF(215.f), COEFF(255.f) }, /* 10700 */
	{ COEFF(202.f), COEFF(214.f), COEFF(255.f) }, /* 10800 */
	{ COEFF(201.f), COEFF(214.f), COEFF(255.f) }, /* 10900 */
	{ COEFF(200.f), COEFF(213.f), COEFF(255.f) }, /* 11000 */
	{ COEFF(200.f), COEFF(213.f), COEFF(255.f) }, /* 11100 */
	{ COEFF(199.f), COEFF(212.f), COEFF(255.f) }, /* 11200 */
	{ COEFF(198.f), COEFF(212.f), COEFF(255.f) }, /* 11300 */
	{ COEFF(198.f), COEFF(212.f), COEFF(255.f) }, /* 11400 */
	{ COEFF(197.f), COEFF(211.f), COEFF(255.f) }, /* 11500 */
	{ COEFF(197.f), COEFF(211.f), COEFF(255.f) }, /* 11600 */
	{ COEFF(196.f), COEFF(210.f), COEFF(255.f) }, /* 11700 */
	{ COEFF(196.f), COEFF(210.f), COEFF(255.f) }, /* 11800 */
	{ COEFF(195.f), COEFF(210.f), COEFF(255.f) }, /* 11900 */
	{ COEFF(195.f), COEFF(209.f), COEFF(255.f) }, /* 12000 */
	{ COEFF(194.f), COEFF(209.f), COEFF(255.f) }, /* 12100 */
	{ COEFF(194.f), COEFF(208.f), COEFF(255.f) }, /* 12200 */
	{ COEFF(193.f), COEFF(208.f), COEFF(255.f) }, /* 12300 */
	{ COEFF(193.f), COEFF(208.f), COEFF(255.f) }, /* 12400 */
	{ COEFF(192.f), COEFF(207.f), COEFF(255.f) }, /* 12500 */
	{ COEFF(192.f), COEFF(207.f), COEFF(255.f) }, /* 12600 */
	{ COEFF(191.f), COEFF(207.f), COEFF(255.f) }, /* 12700 */
	{ COEFF(191.f), COEFF(206.f), COEFF(255.f) }, /* 12800 */
	{ COEFF(190.f), COEFF(206.f), COEFF(255.f) }, /* 12900 */
	{ COEFF(190.f), COEFF(206.f), COEFF(255.f) }  /* 13000 */
};

const float* getColorCoefficients (const int32_t cct) noexcept
{
	if (cct < algoColorTempMin || cct > algoColorTempMax)
		return nullptr;

	const uint32_t idx = (cct - 1000) / 100;
	return cctCoeff[idx];
}


bool rebuildColorCoefficients (rgbCoefficients& cctStruct) noexcept
{
	/* non implemented yet */
	const CCT_TYPE cct = cctStruct.cct;
	const RGB_TYPE tint = cctStruct.tint;

	return true;
}
