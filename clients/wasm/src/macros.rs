//! Declarative macros that cut the per-algorithm boilerplate.
//!
//! - [`batch_model!`] / [`online_model!`] generate a model `resource` wrapper and
//!   its common read surface (`algorithm`, `sorted-scores`, `score`, `top`,
//!   `scores-bulk`, `save-state`; plus `update` for online). An optional
//!   `extras { … }` block splices algorithm-specific accessors (Glicko-2
//!   `players`, Weng-Lin `ratings`) into the same impl.
//! - [`merge_params!`] folds a WIT params record's `option` fields onto the
//!   core's `Default` (scalar fields assigned as-is; `usize` fields cast from
//!   `u32`). Enum/data fields are set by the caller after the merge.

/// A batch model resource (no incremental `update`).
macro_rules! batch_model {
    ($wrapper:ident, $model_trait:path, $core:ty $(, extras { $($extra:item)* })?) => {
        pub struct $wrapper(pub $core);

        impl $model_trait for $wrapper {
            fn algorithm(&self) -> ::std::string::String {
                ::propagon::RankModel::algorithm(&self.0).to_string()
            }
            fn sorted_scores(&self) -> ::std::vec::Vec<(::std::string::String, f64)> {
                $crate::algos::sorted(&self.0)
            }
            fn score(&self, name: ::std::string::String) -> ::core::option::Option<f64> {
                $crate::algos::score_of(&self.0, &name)
            }
            fn top(&self, k: u32) -> ::std::vec::Vec<(::std::string::String, f64)> {
                $crate::algos::top_k(&self.0, k)
            }
            fn scores_bulk(&self) -> $crate::wit::types::ScoresBulk {
                $crate::algos::bulk(&self.0)
            }
            fn save_state(
                &self,
            ) -> ::core::result::Result<::std::string::String, $crate::wit::types::Error> {
                $crate::algos::save(&self.0)
            }
            $($($extra)*)?
        }
    };
}

/// An online model resource: stores the configured algorithm alongside the model
/// so `update` can fold further batches in. `$ds_impl` is the dataset wrapper
/// (a tuple struct over `RefCell<_>`); `$ds_borrow` its generated borrow type.
macro_rules! online_model {
    (
        $wrapper:ident, $model_trait:path, $algo:ty, $core:ty,
        $ds_impl:ty, $ds_borrow:ty
        $(, extras { $($extra:item)* })?
    ) => {
        pub struct $wrapper {
            pub algo: $algo,
            pub model: ::std::cell::RefCell<$core>,
        }

        impl $model_trait for $wrapper {
            fn algorithm(&self) -> ::std::string::String {
                ::propagon::RankModel::algorithm(&*self.model.borrow()).to_string()
            }
            fn sorted_scores(&self) -> ::std::vec::Vec<(::std::string::String, f64)> {
                $crate::algos::sorted(&*self.model.borrow())
            }
            fn score(&self, name: ::std::string::String) -> ::core::option::Option<f64> {
                $crate::algos::score_of(&*self.model.borrow(), &name)
            }
            fn top(&self, k: u32) -> ::std::vec::Vec<(::std::string::String, f64)> {
                $crate::algos::top_k(&*self.model.borrow(), k)
            }
            fn scores_bulk(&self) -> $crate::wit::types::ScoresBulk {
                $crate::algos::bulk(&*self.model.borrow())
            }
            fn save_state(
                &self,
            ) -> ::core::result::Result<::std::string::String, $crate::wit::types::Error> {
                $crate::algos::save(&*self.model.borrow())
            }
            fn update(
                &self,
                data: $ds_borrow,
            ) -> ::core::result::Result<(), $crate::wit::types::Error> {
                let ds = data.get::<$ds_impl>();
                let mut model = self.model.borrow_mut();
                $crate::errors::MapWit::map_wit(::propagon::OnlineRanker::update(
                    &self.algo,
                    &mut model,
                    &ds.0.borrow(),
                ))
            }
            $($($extra)*)?
        }
    };
}

/// Emit a batch algorithm's associated type + `fit`/`fit_warm`/`load` inside an
/// `impl <iface>::Guest for Component` block. `$build` is `fn($params) ->
/// Result<$algo, Error>` (`Ok` for infallible scalar params).
macro_rules! batch_algo {
    (
        $assoc:ident, $wrap:ident, $core:ty, $params:ty, $ds_impl:ty, $ds_borrow:ty,
        $model:ty, $model_borrow:ty, $fit:ident, $warm:ident, $load:ident, $build:path
    ) => {
        type $assoc = $wrap;

        fn $fit(
            params: $params,
            data: $ds_borrow,
        ) -> ::core::result::Result<$model, $crate::wit::types::Error> {
            let algo = $build(params)?;
            let ds = data.get::<$ds_impl>();
            let m =
                $crate::errors::MapWit::map_wit(::propagon::Ranker::fit(&algo, &ds.0.borrow()))?;
            ::core::result::Result::Ok(<$model>::new($wrap(m)))
        }

        fn $warm(
            params: $params,
            data: $ds_borrow,
            init: $model_borrow,
        ) -> ::core::result::Result<$model, $crate::wit::types::Error> {
            let algo = $build(params)?;
            let ds = data.get::<$ds_impl>();
            let init = init.get::<$wrap>();
            let m = $crate::errors::MapWit::map_wit(::propagon::Ranker::fit_warm(
                &algo,
                &ds.0.borrow(),
                &init.0,
            ))?;
            ::core::result::Result::Ok(<$model>::new($wrap(m)))
        }

        fn $load(
            state: ::std::string::String,
        ) -> ::core::result::Result<$model, $crate::wit::types::Error> {
            let m = $crate::errors::MapWit::map_wit(<$core as ::propagon::RankModel>::load_jsonl(
                state.as_bytes(),
            ))?;
            ::core::result::Result::Ok(<$model>::new($wrap(m)))
        }
    };
}

/// Emit a parameterless batch algorithm's associated type + `fit`/`load`.
macro_rules! nofield_algo {
    (
        $assoc:ident, $wrap:ident, $core:ty, $algo:ty, $ds_impl:ty, $ds_borrow:ty,
        $model:ty, $fit:ident, $load:ident
    ) => {
        type $assoc = $wrap;

        fn $fit(data: $ds_borrow) -> ::core::result::Result<$model, $crate::wit::types::Error> {
            let ds = data.get::<$ds_impl>();
            let m = $crate::errors::MapWit::map_wit(::propagon::Ranker::fit(
                &<$algo>::default(),
                &ds.0.borrow(),
            ))?;
            ::core::result::Result::Ok(<$model>::new($wrap(m)))
        }

        fn $load(
            state: ::std::string::String,
        ) -> ::core::result::Result<$model, $crate::wit::types::Error> {
            let m = $crate::errors::MapWit::map_wit(<$core as ::propagon::RankModel>::load_jsonl(
                state.as_bytes(),
            ))?;
            ::core::result::Result::Ok(<$model>::new($wrap(m)))
        }
    };
}

/// Emit an online algorithm's associated type + `init`/`fit`/`load`. `$build` is
/// `fn($params) -> $algo` (infallible). `update` lives on the model resource via
/// [`online_model!`].
macro_rules! online_algo {
    (
        $assoc:ident, $wrap:ident, $core:ty, $algo:ty, $params:ty, $ds_impl:ty,
        $ds_borrow:ty, $model:ty, $init:ident, $fit:ident, $load:ident, $build:path
    ) => {
        type $assoc = $wrap;

        fn $init(params: $params) -> $model {
            let algo = $build(params);
            let model = ::propagon::OnlineRanker::init(&algo);
            <$model>::new($wrap {
                algo,
                model: ::std::cell::RefCell::new(model),
            })
        }

        fn $fit(
            params: $params,
            data: $ds_borrow,
        ) -> ::core::result::Result<$model, $crate::wit::types::Error> {
            let algo = $build(params);
            let ds = data.get::<$ds_impl>();
            let mut model = ::propagon::OnlineRanker::init(&algo);
            $crate::errors::MapWit::map_wit(::propagon::OnlineRanker::update(
                &algo,
                &mut model,
                &ds.0.borrow(),
            ))?;
            ::core::result::Result::Ok(<$model>::new($wrap {
                algo,
                model: ::std::cell::RefCell::new(model),
            }))
        }

        fn $load(
            state: ::std::string::String,
        ) -> ::core::result::Result<$model, $crate::wit::types::Error> {
            let model = $crate::errors::MapWit::map_wit(
                <$core as ::propagon::RankModel>::load_jsonl(state.as_bytes()),
            )?;
            ::core::result::Result::Ok(<$model>::new($wrap {
                algo: <$algo>::default(),
                model: ::std::cell::RefCell::new(model),
            }))
        }
    };
}

/// Fold a params record's `option` fields onto the core algorithm's `Default`.
macro_rules! merge_params {
    (
        $p:expr, $algo:ty
        $(, scalar { $($sf:ident),* $(,)? })?
        $(, usize { $($uf:ident),* $(,)? })?
    ) => {{
        let mut a = <$algo>::default();
        $( $( if let ::core::option::Option::Some(v) = $p.$sf { a.$sf = v; } )* )?
        $( $( if let ::core::option::Option::Some(v) = $p.$uf { a.$uf = v as usize; } )* )?
        a
    }};
}
