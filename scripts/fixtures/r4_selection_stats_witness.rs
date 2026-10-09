#![allow(dead_code)]
#[path = "selection_stats.rs"] mod selection_stats;
use selection_stats::*;
fn probe(name: &str, input: &[f64]) -> (f64,bool,usize) {
 let m=compute_moments(input).unwrap();
 let v=edge_survives_multiplicity(input,100);
 println!("{} input_n={} accepted_n={} sr={:.15} kurtosis={:.15} se={:.15} dsr={:.15} passes={}",name,input.len(),m.n,sharpe(&m),m.kurtosis,sharpe_std_error(&m,sharpe(&m)),v.dsr,v.passes);
 (v.dsr,v.passes,m.n)
}
fn main(){
 let original:Vec<f64>=(0..40).map(|i|if i%2==0 {0.0026}else{-0.0014}).collect();
 let repeated:Vec<f64>=original.iter().flat_map(|x| std::iter::repeat_n(*x,10)).collect();
 let base=probe("original",&original);
 let dup=probe("each_return_repeated_10_times",&repeated);
 assert!(!base.1 && dup.1);
 let mut contaminated=repeated.clone();
 contaminated.extend([f64::NAN,f64::INFINITY,f64::NEG_INFINITY]);
 let dirty=probe("repeated_plus_nan_posinf_neginf",&contaminated);
 assert_eq!(dirty,dup);
 let mut reordered=repeated.clone();
 reordered.sort_by(f64::total_cmp);
 let ordered=probe("repeated_sorted",&reordered);
 assert!((ordered.0-dup.0).abs()<1e-12);
 let mut leptokurtic=vec![0.0;40];
 leptokurtic[0]=0.01;
 let initial=compute_moments(&leptokurtic).unwrap();
 let shift=initial.sd*0.5-initial.mean;
 for x in &mut leptokurtic {*x+=shift;}
 probe("leptokurtic_positive_skew",&leptokurtic);
 let m=compute_moments(&leptokurtic).unwrap();
 let gaussian_zero_sr=1.0/((m.n-1)as f64).sqrt();
 let se=sharpe_std_error(&m,sharpe(&m));
 assert!(m.kurtosis>3.0 && se<gaussian_zero_sr);
 println!("leptokurtic_comparison skew={:.15} se={:.15} baseline_one_over_sqrt_n_minus_one={:.15}",m.skewness,se,gaussian_zero_sr);
 println!("RESULT=PASS characterization assertions; not an OOS result or production pipeline test");
}
