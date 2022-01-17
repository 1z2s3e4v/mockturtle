/* mockturtle: C++ logic network library
 * Copyright (C) 2018-2021  EPFL
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/*!
  \file window_rewriting.hpp
  \brief Window rewriting

  \author Heinz Riener
*/

#include "../networks/events.hpp"
#include "../utils/debugging_utils.hpp"
#include "../utils/index_list.hpp"
#include "../utils/stopwatch.hpp"
#include "../utils/window_utils.hpp"
#include "../views/topo_view.hpp"
#include "../views/window_view.hpp"
#include "../views/fanout_view.hpp"
#include "../views/depth_view.hpp"
#include "../views/color_view.hpp"

#include <abcresub/abcresub2.hpp>
#include <fmt/format.h>
#include <stack>

#include <algorithm>
#include <random>
//#include <vector>

#pragma once

namespace mockturtle
{

struct window_rewriting_params
{
  uint64_t cut_size{6};
  uint64_t num_levels{5};

  /* Level information guides the windowing construction and as such impacts QoR:
     -- dont_update: fastest, but levels are wrong (QoR degrades)
     -- eager: fast, some levels are wrong
     -- precise: fast, all levels are correct (best QoR)
     -- recompute: slow, same as precise (used only for debugging)
  */
  enum
  {
    /* do not update any levels */
    dont_update,
    /* eagerly update the levels of changed nodes but avoid
       topological sorting (some levels will be wrong) */
    eager,
    /* precisely update the levels of changed nodes bottom-to-top and
       in topological order */
    precise,
    /* recompute all levels (also precise, but more expensive to
       compute) */
    recompute,
  } level_update_strategy = dont_update;

  bool filter_cyclic_substitutions{false};
}; /* window_rewriting_params */

struct window_rewriting_stats
{
  /*! \brief Total runtime. */
  stopwatch<>::duration time_total{0};

  /*! \brief Time for constructing windows. */
  stopwatch<>::duration time_window{0};

  /*! \brief Time for optimizing windows. */
  stopwatch<>::duration time_optimize{0};

  /*! \brief Time for substituting. */
  stopwatch<>::duration time_substitute{0};

  /*! \brief Time for updating level information. */
  stopwatch<>::duration time_levels{0};

  /*! \brief Time for topological sorting. */
  stopwatch<>::duration time_topo_sort{0};

  /*! \brief Time for encoding index_list. */
  stopwatch<>::duration time_encode{0};

  /*! \brief Time for detecting cycles. */
  stopwatch<>::duration time_cycle{0};

  /*! \brief Total number of calls to the resub. engine. */
  uint64_t num_substitutions{0};
  uint64_t num_restrashes{0};
  uint64_t num_windows{0};
  uint64_t gain{0};

  window_rewriting_stats operator+=( window_rewriting_stats const& other )
  {
    time_total += other.time_total;
    time_window += other.time_window;
    time_optimize += other.time_optimize;
    time_substitute += other.time_substitute;
    time_levels += other.time_levels;
    time_topo_sort += other.time_topo_sort;
    time_encode += other.time_encode;
    num_substitutions += other.num_substitutions;
    num_restrashes += other.num_restrashes;
    num_windows += other.num_windows;
    gain += other.gain;
    return *this;
  }

  void report() const
  {
    stopwatch<>::duration time_other =
      time_total - time_window - time_topo_sort - time_optimize - time_substitute - time_levels;

    fmt::print( "===========================================================================\n" );
    fmt::print( "[i] Windowing =  {:7.2f} ({:5.2f}%) (#win = {})\n",
                to_seconds( time_window ), to_seconds( time_window ) / to_seconds( time_total ) * 100, num_windows );
    fmt::print( "[i] Top.sort =   {:7.2f} ({:5.2f}%)\n", to_seconds( time_topo_sort ), to_seconds( time_topo_sort ) / to_seconds( time_total ) * 100 );
    fmt::print( "[i] Enc.list =   {:7.2f} ({:5.2f}%)\n", to_seconds( time_encode ), to_seconds( time_encode ) / to_seconds( time_total ) * 100 );
    fmt::print( "[i] Optimize =   {:7.2f} ({:5.2f}%) (#resubs = {}, est. gain = {})\n",
                to_seconds( time_optimize ), to_seconds( time_optimize ) / to_seconds( time_total ) * 100, num_substitutions, gain );
    fmt::print( "[i] Substitute = {:7.2f} ({:5.2f}%) (#hash upd. = {})\n",
                to_seconds( time_substitute ),
                to_seconds( time_substitute ) / to_seconds( time_total ) * 100,
                num_restrashes );
    fmt::print( "[i] Upd.levels = {:7.2f} ({:5.2f}%)\n", to_seconds( time_levels ), to_seconds( time_levels ) / to_seconds( time_total ) * 100 );
    fmt::print( "[i] Other =      {:7.2f} ({:5.2f}%)\n", to_seconds( time_other ), to_seconds( time_other ) / to_seconds( time_total ) * 100 );
    fmt::print( "---------------------------------------------------------------------------\n" );
    fmt::print( "[i] TOTAL =      {:7.2f}\n", to_seconds( time_total ) );
    fmt::print( "===========================================================================\n" );
  }
}; /* window_rewriting_stats */

namespace detail
{

template<typename Ntk>
bool is_contained_in_tfi_recursive( Ntk const& ntk, typename Ntk::node const& node, typename Ntk::node const& n )
{
  if ( ntk.color( node ) == ntk.current_color() )
  {
    return false;
  }
  ntk.paint( node );

  if ( n == node )
  {
    return true;
  }

  bool found = false;
  ntk.foreach_fanin( node, [&]( typename Ntk::signal const& fi ){
    if ( is_contained_in_tfi_recursive( ntk, ntk.get_node( fi ), n ) )
    {
      found = true;
      return false;
    }
    return true;
  });

  return found;
}

} /* namespace detail */

template<typename Ntk>
bool is_contained_in_tfi( Ntk const& ntk, typename Ntk::node const& node, typename Ntk::node const& n )
{
  /* do not even build the TFI, but just search for the node */
  ntk.new_color();
  return detail::is_contained_in_tfi_recursive( ntk, node, n );
}

namespace detail
{

template<class Ntk>
class window_rewriting_impl
{
public:
  using node = typename Ntk::node;
  using signal = typename Ntk::signal;

  struct substitution_data
  {
    substitution_data(
      const uint32_t _n,
      const std::optional<mockturtle::abc_index_list> _il_opt,
      const std::vector<signal> _signals,
      const std::vector<signal> _outputs
    ): n(_n), il_opt(_il_opt), signals(_signals), outputs(_outputs) {}

    uint32_t n;
    std::optional<mockturtle::abc_index_list> il_opt;
    std::vector<signal> signals;
    std::vector<signal> outputs;
  }; /* substitution_data */

public:
  explicit window_rewriting_impl( Ntk& ntk, window_rewriting_params const& ps, window_rewriting_stats& st )
    : ntk( ntk )
    , ps( ps )
    , st( st )
    /* initialize levels to network depth */
    , levels( ntk.depth() )
  {
    register_events();
  }

  ~window_rewriting_impl()
  {
    ntk.events().release_add_event( add_event );
    ntk.events().release_modified_event( modified_event );
    ntk.events().release_delete_event( delete_event );
  }

  void run()
  {
    stopwatch t( st.time_total );

    /// construct the create_window_impl with ntk
    create_window_impl windowing( ntk );

    uint32_t current_max_gain = 0u;
    uint32_t const size = ntk.size();

    // std::vector<substitution_data> vSubData;
    std::vector<uint32_t> vWindowGain;
    vWindowGain.reserve(size);

    // sort nodes
    std::vector<uint32_t> sorted_n;
    for ( uint32_t n = 0u; n < size; ++n ){
      sorted_n.push_back(n);
    }

    // >>> topo sort n
    // topo_view topo_ntk{ntk};
    // topo_ntk.foreach_node([&]( auto n ) {
    //   sorted_n.push_back(n);
    // });
    // std::reverse(sorted_n.begin(), sorted_n.end());
    // <<< topo sort n 

    // >>> shuffle sort n 
    // auto rng = std::default_random_engine {};
    // sstd::shuffle(std::begin(sorted_n), std::end(sorted_n), rng);
    // <<< shuffle sort n 

    // >>> div num sort n
    // std::vector<int> n2divNum;
    // for ( uint32_t n_i = 0u; n_i < size; ++n_i ){
    //   uint32_t n = sorted_n[n_i];
    //   if ( ntk.is_constant( n ) || ntk.is_ci( n ) || ntk.is_dead( n ) ){
    //     n2divNum.push_back(-1);
    //     continue;
    //   }
    //   if ( const auto w = call_with_stopwatch( st.time_window, [&]() { return windowing.run( n, ps.cut_size, ps.num_levels ); } ) ){
    //     ++st.num_windows;
    //     auto topo_win = call_with_stopwatch( st.time_topo_sort, ( [&](){
    //       window_view win( ntk, w->inputs, w->outputs, w->nodes );
    //       topo_view topo_win{win};
    //       return topo_win;
    //     }) );
    //     abc_index_list il;
    //     call_with_stopwatch( st.time_encode, [&]() {
    //       encode( il, topo_win );
    //     } );
    //     n2divNum.push_back( get_div_num( il ) );
    //   }
    // } 
    // std::sort(sorted_n.begin(),sorted_n.end(), [&](const int n_l, const int n_r){
    //   return n2divNum[n_l] > n2divNum[n_r];
    // });
    // <<< div num sort n

    // >>> sort gain
    for ( uint32_t n_i = 0u; n_i < size; ++n_i ){
      uint32_t n = sorted_n[n_i];
      if ( ntk.is_constant( n ) || ntk.is_ci( n ) || ntk.is_dead( n ) )
      {
        vWindowGain.emplace_back(0);
        continue;
      }

      if ( const auto w = call_with_stopwatch( st.time_window, [&]() { return windowing.run( n, ps.cut_size, ps.num_levels ); } ) )
      {
        ++st.num_windows;

        auto topo_win = call_with_stopwatch( st.time_topo_sort, ( [&](){
          window_view win( ntk, w->inputs, w->outputs, w->nodes );
          topo_view topo_win{win};
          return topo_win;
        }) );
        abc_index_list il;
        call_with_stopwatch( st.time_encode, [&]() {
          encode( il, topo_win );
        } );

        auto il_opt = optimize( il );
        if ( !il_opt )
        {
          vWindowGain.emplace_back(0);
          continue;
        }
        vWindowGain.emplace_back(il.num_gates() - il_opt->num_gates());
      }
      else { vWindowGain.emplace_back(0); }
    }

    sort(sorted_n.begin(), sorted_n.end(),
      [&](const uint32_t& a, const uint32_t& b) -> bool {
        return vWindowGain[a] > vWindowGain[b];
      }
    );
    uint32_t num_zeros = std::count(vWindowGain.begin(), vWindowGain.end(), 0);
    uint32_t sort_gain_threshold = 0;

    /*
    std::cout << "gain threshold = " << sort_gain_threshold << std::endl;
    for ( uint32_t n = 0u; n < size; ++n ){
      if(vWindowGain[sorted_n[n]] != 0)
        std::cout << "node (" << std::setw(5) << sorted_n[n] << ") with gain = " << vWindowGain[sorted_n[n]] << std::endl;
    }
    */
    // <<< sort gain

    /// Algorithm M : for each node run Algorithm S
    for ( uint32_t n_i = 0u; n_i < size; ++n_i )
    {
      uint32_t n = sorted_n[n_i];
      //printf("node[%d]\n", n);
      if ( ntk.is_constant( n ) || ntk.is_ci( n ) || ntk.is_dead( n ) )
      {
        //printf("node %d is constant/ci/dead\n",n);
        continue;
      }
      
      // >>> sort gain
      if ( vWindowGain[n] < sort_gain_threshold || vWindowGain[n] < 1 ) continue;
      // <<< sort gain

      /// Algorithm W : Create window w with pivot n. ( W1~6 in windowing.run() )
      if ( const auto w = call_with_stopwatch( st.time_window, [&]() { return windowing.run( n, ps.cut_size, ps.num_levels ); } ) )
      {
        /// w = window{fins, *nodes, fouts}
        ++st.num_windows;

        /// window --> topo
        auto topo_win = call_with_stopwatch( st.time_topo_sort, ( [&](){
          window_view win( ntk, w->inputs, w->outputs, w->nodes );
          topo_view topo_win{win};
          return topo_win;
        }) );
        abc_index_list il;
        call_with_stopwatch( st.time_encode, [&]() {
          encode( il, topo_win );
        } );

        /// (Algorithm M+S) Optimize the window(topo) with Abc_ResubComputeWindow
        auto il_opt = optimize( il );
        if ( !il_opt )
        {
          continue;
        }

        // >>> non-desending gain
        // if ( il.num_gates() - il_opt->num_gates() < current_max_gain) continue;
        // current_max_gain = il.num_gates() - il_opt->num_gates();
        // std::cout << "[t] node(" << n << ") il size difference = " << il.num_gates() - il_opt->num_gates() << std::endl;
        // <<< non-desending gain

        /// Resyn finished. Get fins and fouts of window for replace ntk
        std::vector<signal> signals;
        for ( auto const& i : w->inputs )
        {
          signals.push_back( ntk.make_signal( i ) );
        }
        std::vector<signal> outputs;
        topo_win.foreach_co( [&]( signal const& o ){
          outputs.push_back( o );
        });

        // vSubData.emplace_back(n, il_opt, signals, outputs);

        uint32_t counter{0};
        ++st.num_substitutions;

        /* ensure that no dead nodes are reachable */
        assert( count_reachable_dead_nodes( ntk ) == 0u );

        std::list<std::pair<node, signal>> substitutions;
        insert( ntk, std::begin( signals ), std::end( signals ), *il_opt,
                [&]( signal const& _new )
                {
                  assert( !ntk.is_dead( ntk.get_node( _new ) ) );
                  auto const _old = outputs.at( counter++ );
                  if ( _old == _new )
                  {
                    return true;
                  }

                  /* ensure that _old is not in the TFI of _new */
                  // assert( !is_contained_in_tfi( ntk, ntk.get_node( _new ), ntk.get_node( _old ) ) );
                  if ( ps.filter_cyclic_substitutions &&
                       call_with_stopwatch( st.time_window, [&](){ return is_contained_in_tfi( ntk, ntk.get_node( _new ), ntk.get_node( _old ) ); }) )
                  {
                    std::cout << "undo resubstitution " << ntk.get_node( _old ) << std::endl;
                    substitutions.emplace_back( std::make_pair( ntk.get_node( _old ), ntk.is_complemented( _old ) ? !_new : _new ) );                    
                    for ( auto it = std::rbegin( substitutions ); it != std::rend( substitutions ); ++it )
                    {
                      if ( ntk.fanout_size( ntk.get_node( it->second ) ) == 0u )
                      {
                        ntk.take_out_node( ntk.get_node( it->second ) );
                      }
                    }
                    substitutions.clear();
                    return false;
                  }

                  substitutions.emplace_back( std::make_pair( ntk.get_node( _old ), ntk.is_complemented( _old ) ? !_new : _new ) );
                  return true;
                });

        /* ensure that no dead nodes are reachable */
        assert( count_reachable_dead_nodes( ntk ) == 0u );
        substitute_nodes( substitutions );

        /* recompute levels and depth */
        if ( ps.level_update_strategy == window_rewriting_params::recompute )
        {
          call_with_stopwatch( st.time_levels, [&]() { ntk.update_levels(); } );
        }
        if ( ps.level_update_strategy != window_rewriting_params::dont_update )
        {
          update_depth();
        }

        /* ensure that no dead nodes are reachable */
        assert( count_reachable_dead_nodes( ntk ) == 0u );

        /* ensure that the network structure is still acyclic */
        assert( network_is_acylic( ntk ) );

        if ( ps.level_update_strategy == window_rewriting_params::precise ||
             ps.level_update_strategy == window_rewriting_params::recompute )
        {
          /* ensure that the levels and depth is correct */
          assert( check_network_levels( ntk ) );
        }

        /* update internal data structures in windowing */
        windowing.resize( ntk.size() );
      }
    }
/*
    for (auto data : vSubData)
    {
        std::optional<mockturtle::abc_index_list> il_opt = data.il_opt;
        std::vector<signal> signals = data.signals;
        std::vector<signal> outputs = data.outputs;
    }
  */

    /* ensure that no dead nodes are reachable */
    assert( count_reachable_dead_nodes( ntk ) == 0u );
  }

private:
  void register_events()
  {
    auto const update_level_of_new_node = [&]( const auto& n ) {
      stopwatch t( st.time_total );
      update_levels( n );
    };

    auto const update_level_of_existing_node = [&]( node const& n, const auto& old_children ) {
      (void)old_children;
      stopwatch t( st.time_total );
      update_levels( n );
    };

    auto const update_level_of_deleted_node = [&]( node const& n ) {
      stopwatch t( st.time_total );
      assert( ntk.fanout_size( n ) == 0u );
      assert( ntk.is_dead( n ) );
      ntk.set_level( n, -1 );
    };

    add_event = ntk.events().register_add_event( update_level_of_new_node );
    modified_event = ntk.events().register_modified_event( update_level_of_existing_node );
    delete_event = ntk.events().register_delete_event( update_level_of_deleted_node );
  }

  int get_div_num( abc_index_list const& il, bool verbose = false ){
    int nDivNum = 0;
    stopwatch t( st.time_optimize );
    int *raw = ABC_CALLOC( int, il.size() + 1u );
    uint64_t i = 0;
    for ( auto const& v : il.raw() ){
      raw[i++] = v;
    }
    raw[1] = 0; /* fix encoding */
 
    abcresub::Abc_ResubPrepareManager( 1 );
    int *new_raw = nullptr;
    int num_resubs = 0;
    nDivNum = abcresub::Abc_ResubComputeWindow_getDivNum( raw, ( il.size() / 2u ), 1000, -1, 0, 0, 0, 0, &new_raw, &num_resubs );
    abcresub::Abc_ResubPrepareManager( 0 );

    if ( raw ){
      ABC_FREE( raw );
      return nDivNum;
    }else {
      return -1;
    }
  }

  /* optimize an index_list and return the new list */
  std::optional<abc_index_list> optimize( abc_index_list const& il, bool verbose = false )
  {
    stopwatch t( st.time_optimize );

    int *raw = ABC_CALLOC( int, il.size() + 1u );
    uint64_t i = 0;
    for ( auto const& v : il.raw() )
    {
      raw[i++] = v;
    }
    raw[1] = 0; /* fix encoding */

    /// >>> Algorithm M : given a window il, resynthesizes the outputs and replace them 
    abcresub::Abc_ResubPrepareManager( 1 );
    int *new_raw = nullptr;
    int num_resubs = 0;
    uint64_t new_entries = abcresub::Abc_ResubComputeWindow( raw, ( il.size() / 2u ), 1000, -1, 0, 0, 0, 0, &new_raw, &num_resubs );
    abcresub::Abc_ResubPrepareManager( 0 );
    /// <<< Algorithm M : given a window il, resynthesizes the outputs and replace them 

    if ( verbose )
    {
      fmt::print( "Performed resub {} times.  Reduced {} nodes.\n",
                  num_resubs, new_entries > 0 ? ( ( il.size() / 2u ) - new_entries ) : 0 );
    }
    st.gain += new_entries > 0 ? ( ( il.size() / 2u ) - new_entries ) : 0;

    if ( raw )
    {
      ABC_FREE( raw );
    }

    if ( new_entries > 0 )
    {
      std::vector<uint32_t> values;
      for ( uint32_t i = 0; i < 2*new_entries; ++i )
      {
        values.push_back( new_raw[i] );
      }
      values[1u] = 1; /* fix encoding */
      if ( new_raw )
      {
        ABC_FREE( new_raw );
      }
      return abc_index_list( values, il.num_pis() );
    }
    else
    {
      assert( new_raw == nullptr );
      return std::nullopt;
    }
  }

  void substitute_nodes( std::list<std::pair<node, signal>> substitutions )
  {
    stopwatch t( st.time_substitute );

    auto clean_substitutions = [&]( node const& n )
    {
      substitutions.erase( std::remove_if( std::begin( substitutions ), std::end( substitutions ),
                                           [&]( auto const& s ){
                                             if ( s.first == n )
                                             {
                                               node const nn = ntk.get_node( s.second );
                                               if ( ntk.is_dead( nn ) )
                                                 return true;

                                               /* deref fanout_size of the node */
                                               if ( ntk.fanout_size( nn ) > 0 )
                                               {
                                                 ntk.decr_fanout_size( nn );
                                               }
                                               /* remove the node if its fanout_size becomes 0 */
                                               if ( ntk.fanout_size( nn ) == 0 )
                                               {
                                                 ntk.take_out_node( nn );
                                               }
                                               /* remove substitution from list */
                                               return true;
                                             }
                                             return false; /* keep */
                                           } ),
                           std::end( substitutions ) );
    };

    /* register event to delete substitutions if their right-hand side
       nodes get deleted */
    auto clean_subs_event = ntk.events().register_delete_event( clean_substitutions );

    /* increment fanout_size of all signals to be used in
       substitutions to ensure that they will not be deleted */
    for ( const auto& s : substitutions )
    {
      ntk.incr_fanout_size( ntk.get_node( s.second ) );
    }

    while ( !substitutions.empty() )
    {
      auto const [old_node, new_signal] = substitutions.front();
      substitutions.pop_front();

      for ( auto index : ntk.fanout( old_node ) )
      {
        /* skip CIs and dead nodes */
        if ( ntk.is_dead( index ) )
        {
          continue;
        }

        /* skip nodes that will be deleted */
        if ( std::find_if( std::begin( substitutions ), std::end( substitutions ),
                           [&index]( auto s ){ return s.first == index; } ) != std::end( substitutions ) )
        {
          continue;
        }

        /* replace in node */
        if ( const auto repl = ntk.replace_in_node( index, old_node, new_signal ); repl )
        {
          ntk.incr_fanout_size( ntk.get_node( repl->second ) );
          substitutions.emplace_back( *repl );
          ++st.num_restrashes;
        }
      }

      /* replace in outputs */
      ntk.replace_in_outputs( old_node, new_signal );

      /* replace in substitutions */
      for ( auto& s : substitutions )
      {
        if ( ntk.get_node( s.second ) == old_node )
        {
          s.second = ntk.is_complemented( s.second ) ? !new_signal : new_signal;
          ntk.incr_fanout_size( ntk.get_node( new_signal ) );
        }
      }

      /* finally remove the node: note that we never decrement the
         fanout_size of the old_node. instead, we remove the node and
         reset its fanout_size to 0 knowing that it must be 0 after
         substituting all references. */
      assert( !ntk.is_dead( old_node ) );
      ntk.take_out_node( old_node );

      /* decrement fanout_size when released from substitution list */
      ntk.decr_fanout_size( ntk.get_node( new_signal ) );
      if ( ntk.fanout_size( ntk.get_node( new_signal ) ) == 0 )
      {
        ntk.take_out_node( ntk.get_node( new_signal ) );
      }
    }

    ntk.events().release_delete_event( clean_subs_event );
  }

  void update_levels( node const& n )
  {
    ntk.resize_levels();
    if ( ps.level_update_strategy == window_rewriting_params::precise )
    {
      call_with_stopwatch( st.time_levels, [&]() { update_node_level_precise( n ); } );
    }
    else if ( ps.level_update_strategy == window_rewriting_params::eager )
    {
      call_with_stopwatch( st.time_levels, [&]() { update_node_level_eager( n ); } );
    }

    /* levels can be wrong until substitute_nodes has finished */
    // assert( check_network_levels( ntk ) );
  }

  /* precisely update node levels using an iterative topological sorting approach */
  void update_node_level_precise( node const& n )
  {
    assert( count_reachable_dead_nodes_from_node( ntk, n ) == 0u );
    // assert( count_nodes_with_dead_fanins( ntk, n ) == 0u );

    /* compute level of current node */
    uint32_t level_offset{0};
    ntk.foreach_fanin( n, [&]( signal const& fi ){
      level_offset = std::max( ntk.level( ntk.get_node( fi ) ), level_offset );
    });
    ++level_offset;

    /* add node into levels */
    if ( levels.size() < 1u )
    {
      levels.resize( 1u );
    }
    levels[0].emplace_back( n );

    for ( uint32_t level_index = 0u; level_index < levels.size(); ++level_index )
    {
      if ( levels[level_index].empty() )
        continue;

      for ( uint32_t node_index = 0u; node_index < levels[level_index].size(); ++node_index )
      {
        node const p = levels[level_index][node_index];

        /* recompute level of this node */
        uint32_t lvl{0};
        ntk.foreach_fanin( p, [&]( signal const& fi ){
          if ( ntk.is_dead( ntk.get_node( fi ) ) )
            return;

          lvl = std::max( ntk.level( ntk.get_node( fi ) ), lvl );
          return;
        });
        ++lvl;
        assert( lvl > 0 );

        /* update level and add fanouts to levels[.] if the recomputed
           level is different from the current level */
        if ( lvl != ntk.level( p ) )
        {
          ntk.set_level( p, lvl );
          ntk.foreach_fanout( p, [&]( node const& fo ){
            assert( std::max( ntk.level( fo ), lvl + 1 ) >= level_offset );
            uint32_t const pos = std::max( ntk.level( fo ), lvl + 1 ) - level_offset;
            assert( pos >= 0u );
            assert( pos >= level_index );
            if ( levels.size() <= pos )
            {
              levels.resize( std::max( uint32_t( levels.size() << 1 ), pos + 1 ) );
            }
            levels[pos].emplace_back( fo );
          });
        }
      }

      /* clean the level */
      levels[level_index].clear();
    }
    levels.clear();
  }

  /* eagerly update the node levels without topologically sorting (may
     stack-overflow if the network is deep)*/
  void update_node_level_eager( node const& n )
  {
    uint32_t const curr_level = ntk.level( n );
    uint32_t max_level = 0;
    ntk.foreach_fanin( n, [&]( const auto& f ) {
      auto const p = ntk.get_node( f );
      auto const fanin_level = ntk.level( p );
      if ( fanin_level > max_level )
      {
        max_level = fanin_level;
      }
    } );
    ++max_level;

    if ( curr_level != max_level )
    {
      ntk.set_level( n, max_level );
      ntk.foreach_fanout( n, [&]( const auto& p ) {
        if ( !ntk.is_dead( p ) )
        {
          update_node_level_eager( p );
        }
      } );
    }
  }

  /* update network depth (needs level information!) */
  void update_depth()
  {
    stopwatch t( st.time_levels );

    uint32_t max_level{0};
    ntk.foreach_co( [&]( signal const& s ){
      assert( !ntk.is_dead( ntk.get_node( s ) ) );
      max_level = std::max( ntk.level( ntk.get_node( s ) ), max_level );
    });

    if ( ntk.depth() != max_level )
    {
      ntk.set_depth( max_level );
    }
  }

private:
  Ntk& ntk;
  window_rewriting_params ps;
  window_rewriting_stats& st;

  std::vector<std::vector<node>> levels;

  /* events */
  std::shared_ptr<typename network_events<Ntk>::add_event_type> add_event;
  std::shared_ptr<typename network_events<Ntk>::modified_event_type> modified_event;
  std::shared_ptr<typename network_events<Ntk>::delete_event_type> delete_event;
};

} /* namespace detail */

template<class Ntk>
void window_rewriting( Ntk& ntk, window_rewriting_params const& ps = {}, window_rewriting_stats* pst = nullptr )
{
  fanout_view fntk{ntk};
  depth_view dntk{fntk};
  color_view cntk{dntk};

  window_rewriting_stats st;
  detail::window_rewriting_impl p( cntk, ps, st );
  p.run();
  if ( pst )
  {
    *pst = st;
  }
}

} /* namespace mockturtle */
