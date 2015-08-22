!=========== twohalo.f90 =======================================================
!
!  Perform the integration for the two-halo term, along with halo exclusion.
!
!  Routines:
!      * spline [calculate cubic spline interpolation]
!      * ispline [evaluate spline at input ordinate]
!      * power_to_corr [perform hankel transform using ogata05 method]
!      * power_to_corr_matrix [do power_to_corr for several scales efficiently]
!      * scale_dep_bias [Tinker05 scale-dependent bias]
!      * virial_mass [calculate virial mass from virial radius
!      * ng_dash [calculate modified mean density (below mlim)
!      * overlapping_halo_prob [probability of overlapping triaxial halos from Tinker+05]
!      * virial_radius [inverse of virial_mass]
!      * twohalo [THE MAIN CALCULATION ROUTINE -- calculates the 2-halo term]
!      * dblsimps [2-dimensional simpsons integration]
!      * simps [1-d simpsons intgration]
!
!  TODO: Clean up and organise the code better.
!        The halo exclusion and scale-dependent bias models are NOT very flexible,
!        unlike all other models in the code. THIS SHOULD BE DEALT WITH.
!===============================================================================

module twohalo_calc
!    use hankel
!    use utils
    implicit none

!    real(8), parameter :: pi=3.141592653589

    real(8), parameter :: pi=3.141592653589

    integer :: myN = 640
    real(8) :: myh = 0.005

    contains

    subroutine spline (x, y, b, c, d, n)
    !======================================================================
    !  Calculate the coefficients b(i), c(i), and d(i), i=1,2,...,n
    !  for cubic spline interpolation
    !  s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3
    !  for  x(i) <= x <= x(i+1)
    !  Alex G: January 2010
    !----------------------------------------------------------------------
    !  input..
    !  x = the arrays of data abscissas (in strictly increasing order)
    !  y = the arrays of data ordinates
    !  n = size of the arrays xi() and yi() (n>=2)
    !  output..
    !  b, c, d  = arrays of spline coefficients
    !  comments ...
    !  spline.f90 program is based on fortran version of program spline.f
    !  the accompanying function fspline can be used for interpolation
    !======================================================================
        implicit none

        integer, intent(in) :: n
        real(8), intent(in) :: x(n), y(n)
        real(8), intent(out):: b(n), c(n), d(n)

        integer :: i, j, gap
        real(8) :: h

        gap = n-1

        ! check input
        if ( n < 2 ) return
        if ( n < 3 ) then
          b(1) = (y(2)-y(1))/(x(2)-x(1))   ! linear interpolation
          c(1) = 0.
          d(1) = 0.
          b(2) = b(1)
          c(2) = 0.
          d(2) = 0.
          return
        end if
        !
        ! step 1: preparation
        !
        d(1) = x(2) - x(1)
        c(2) = (y(2) - y(1))/d(1)
        do i = 2, gap
          d(i) = x(i+1) - x(i)
          b(i) = 2.0*(d(i-1) + d(i))
          c(i+1) = (y(i+1) - y(i))/d(i)
          c(i) = c(i+1) - c(i)
        end do
        !
        ! step 2: end conditions
        !
        b(1) = -d(1)
        b(n) = -d(n-1)
        c(1) = 0.0
        c(n) = 0.0
        if(n /= 3) then
          c(1) = c(3)/(x(4)-x(2)) - c(2)/(x(3)-x(1))
          c(n) = c(n-1)/(x(n)-x(n-2)) - c(n-2)/(x(n-1)-x(n-3))
          c(1) = c(1)*d(1)**2/(x(4)-x(1))
          c(n) = -c(n)*d(n-1)**2/(x(n)-x(n-3))
        end if
        !
        ! step 3: forward elimination
        !
        do i = 2, n
          h = d(i-1)/b(i-1)
          b(i) = b(i) - h*d(i-1)
          c(i) = c(i) - h*c(i-1)
        end do
        !
        ! step 4: back substitution
        !
        c(n) = c(n)/b(n)
        do j = 1, gap
          i = n-j
          c(i) = (c(i) - d(i)*c(i+1))/b(i)
        end do
        !
        ! step 5: compute spline coefficients
        !
        b(n) = (y(n) - y(gap))/d(gap) + d(gap)*(c(gap) + 2.0*c(n))
        do i = 1, gap
          b(i) = (y(i+1) - y(i))/d(i) - d(i)*(c(i+1) + 2.0*c(i))
          d(i) = (c(i+1) - c(i))/d(i)
          c(i) = 3.*c(i)
        end do
        c(n) = 3.0*c(n)
        d(n) = d(n-1)
    end subroutine spline

    function ispline(u, x, y, b, c, d, n)
    !======================================================================
    ! function ispline evaluates the cubic spline interpolation at point z
    ! ispline = y(i)+b(i)*(u-x(i))+c(i)*(u-x(i))**2+d(i)*(u-x(i))**3
    ! where  x(i) <= u <= x(i+1)
    !----------------------------------------------------------------------
    ! input..
    ! u       = the abscissa at which the spline is to be evaluated
    ! x, y    = the arrays of given data points
    ! b, c, d = arrays of spline coefficients computed by spline
    ! n       = the number of data points
    ! output:
    ! ispline = interpolated value at point u
    !=======================================================================
        implicit none
        real(8) :: ispline
        integer, intent(in) :: n
        real(8), intent(in) :: u, x(n), y(n), b(n), c(n), d(n)

        integer :: i, j, k
        real(8) :: dx

        ! if u is ouside the x() interval take a boundary value (left or right)
        if(u <= x(1)) then
          ispline = y(1)
          return
        end if
        if(u >= x(n)) then
          ispline = y(n)
          return
        end if

        !*
        !  binary search for for i, such that x(i) <= u <= x(i+1)
        !*
        !i = 1
        !j = n+1
        !do while (j > i+1)
        !  k = (i+j)/2
        !  if(u < x(k)) then
        !    j=k
        !    else
        !    i=k
        !   end if
        !end do

        !If data is evenly spaced, can do the following
        dx = x(2)-x(1)
        i = int(ceiling((u-x(1))/dx))
        !*
        !  evaluate spline interpolation
        !*
        dx = u - x(i)
        ispline = y(i) + dx*(b(i) + dx*(c(i) + dx*d(i)))
    end function ispline

    subroutine power_to_corr(nr,nk,r,power,lnk,N,h,corr)
        !=======================================================================
        ! Use Ogata's method for Hankel Transforms in 3D for nu=0 (nu=1/2 for 2D)
        ! to convert a given power spectrum to a correlation function.
        ! Note, in its current form, lnk must be evenly-spaced.
        !=======================================================================
        integer, intent(in) :: nr, nk  ! Lengths of r,lnk respectively
        real(8), intent(in) :: r(nr)   ! The scales at which to get the correlation
        real(8), intent(in) :: power(nk), lnk(nk) ! The power spectrum values and corresponding ln(k) values
        integer, intent(in) :: N       ! Number of steps in integration (~500 is good)
        real(8), intent(in) :: h       ! Step-size of integration (~0.0044 is good)
        real(8), intent(out) :: corr(nr) !The output correlation

        real(8) :: b(nk),c(nk),d(nk)
        real(8) :: sumparts(N), t(N), s(N), x(N),dpsi(N),roots(N),allparts(N)
        integer :: i,j

        call spline(lnk,power,b,c,d,size(lnk))

        do i=1,N
            roots(i) = i*1.d0
        end do

        t = h*roots
        s = pi*sinh(t)
        x = pi * roots * tanh(s/2)
        dpsi = 1.d0+cosh(s)
        where (dpsi.ne.0.d0) dpsi = (pi*t*cosh(t)+sinh(s))/dpsi
        sumparts = pi*sin(x)*dpsi*x
        do i=1,nr
            do j=1,N
                allparts(j) = sumparts(j) * ispline(log(x(j)/r(i)), lnk,power, b, c, d, size(lnk))
            end do
            corr(i) = sum(allparts)/(2*pi**2*r(i)**3)
        end do
    end subroutine

    subroutine power_to_corr_matrix(nr,nk,r,power,lnk,N,h,corr)
        integer, intent(in) :: nr, nk
        real(8), intent(in) :: r(nr), power(nk,nr), lnk(nk)
        integer, intent(in) :: N
        real(8), intent(in) :: h
        real(8), intent(out) :: corr(nr)

        real(8) :: b(nk),c(nk),d(nk)
        real(8) :: sumparts(N), t(N), s(N), x(N),dpsi(N),roots(N),allparts(N)
        integer :: i,j



        do i=1,N
            roots(i) = i*1.d0
        end do

        t = h*roots
        s = pi*sinh(t)
        x = pi * roots * tanh(s/2)
        dpsi = 1.d0+cosh(s)
        where (dpsi.ne.0.d0) dpsi = (pi*t*cosh(t)+sinh(s))/dpsi
        sumparts = pi*sin(x)*dpsi*x
        do i=1,nr
            call spline(lnk,power(:,i),b,c,d,size(lnk))
            do j=1,N
                allparts(j) = sumparts(j) * ispline(log(x(j)/r(i)), lnk,power(:,i), b, c, d, size(lnk))
            end do
            corr(i) = sum(allparts)/(2*pi**2*r(i)**3)
        end do
    end subroutine

    function scale_dep_bias(xi)

        real(8), intent(in) :: xi
        real(8) :: scale_dep_bias

        scale_dep_bias = sqrt((1.d0 + 1.17d0 * xi) ** 1.49d0 / (1.d0 + 0.69d0 * xi) ** 2.09d0)
    end function

    function virial_mass(r,mean_dens,delta_halo)

        real(8), intent(in) :: r,mean_dens,delta_halo
        real(8) :: virial_mass
        virial_mass = 4 * pi * r ** 3 * mean_dens * delta_halo / 3
    end function virial_mass

    function ng_dash(m,ntot,dndm,mlim)
        real(8), intent(in) :: m(:), ntot(:), dndm(:),mlim
        real(8) :: ng_dash
        real(8) :: integrand(size(m))

        integrand = 0.d0
        where (m<mlim) integrand = m*ntot*dndm
        if (mlim-m(1)>0.d0)then
            call simps(log(m(2))-log(m(1)),integrand,ng_dash)
        else
            ng_dash=0.d0
        end if
    end function

    subroutine overlapping_halo_prob(r,n1,n2,outersum,res)
        integer, intent(in) :: n1,n2
        real(8), intent(in) :: r, outersum(n1,n2)
        real(8), intent(out):: res(:,:)

        real(8) :: y(n1,n2),x(n1,n2)

        x = r/outersum
        y = (x - 0.8) / 0.29


        res = 3 * y ** 2 - 2 * y ** 3
        where (y.le.0.d0) res = 0.d0
        where (y.ge.1.d0) res = 1.d0

    end subroutine overlapping_halo_prob

    subroutine virial_radius(m,mean_dens,dhalo,out)

        real(8), intent(in) :: m(:), mean_dens,dhalo
        real(8), intent(out):: out(:)

        out = ((3 * m) / (4 * pi * mean_dens * dhalo)) ** (1. / 3.)
    end subroutine virial_radius

    subroutine twohalo(nm,nk,nr,m,bias,ntot,dndm,lnk,dmpower,u,r,dmcorr,nbar,&
                       &dhalo,rhob,exc_type,scale_dep,n_cores,corr)

        !!! INPUT/OUTPUT VARIABLES !!!
        integer, intent(in) :: nm,nk,nr
        real(8), intent(in) :: m(nm),bias(nm), ntot(nm), dndm(nm)
        real(8), intent(in) :: u(nm,nk),dmpower(nk),lnk(nk)
        real(8), intent(in) :: r(nr),dmcorr(nr)
        real(8), intent(in) :: nbar,dhalo,rhob
        integer, intent(in) :: exc_type !Type of exclusion (1,2,3,4,5)
        logical, intent(in) :: scale_dep !Whether it is scale-dependent
        integer, intent(in) :: n_cores
        real(8), intent(out):: corr(nr)

        !!! VARIABLES NEEDED FOR EVERYTHING !!!
        real(8) :: integrand_m(nm), integrandm(nm),pg2h(nk,nr), integrand(nm),dlogm
        real(8) :: ng_integrand(nm)
        integer :: i,j, k

        !!! VARIABLES FOR NG-MATCHED AND ELLIPSOIDAL !!!
        real(8) :: rvir(nm),ngdash(nr),mlim,val,preval
        real(8), allocatable :: prob(:,:), outersum(:,:), integprod(:,:),integprod2(:,:)
        integer :: final_index

        !!! VARIABLES FOR SCHNEIDER !!!
        real(8),allocatable :: x(:), window(:)

        ! General quantities
        !integrand = 0.d0
        dlogm = (log(m(2))-log(m(1)))
        ng_integrand = m*dndm * ntot
        if(.not.scale_dep) integrandm = ng_integrand * bias

        !Type-dependent setup
        if(exc_type==5.or.exc_type==4)then
            call virial_radius(m,rhob,dhalo,rvir)
            allocate(outersum(nm,nm),integprod(nm,nm),integprod2(nm,nm))
            do i=1,nm
                outersum(:,i) = rvir(i) + rvir
            end do

            allocate(prob(nm,nm))
            do i=1,nm
                integprod(:,i) =  ng_integrand(i)*ng_integrand
            end do
        elseif(exc_type==2)then !Schneider
            allocate(x(nk),window(nk))
            x = exp(lnk)*3.d0
            window = (3 * (sin(x) - x * cos(x)) / x ** 3)
        endif

        !$OMP parallel do private(integrand_m,prob,val,i,final_index,mlim,integrand,k,integprod2) NUM_THREADS(n_cores)
        do j=1,nr
            integrand = 0.d0
            if (scale_dep)then
                integrand_m = ng_integrand * scale_dep_bias(dmcorr(j))*bias
            else
                integrand_m = integrandm
            end if

            if(exc_type==5.or.exc_type==4)then !! Ellipsoidal
                call overlapping_halo_prob(r(j),nm,nm,outersum,prob)
                prob = prob * integprod
                call dblsimps(prob,nm,nm,dlogm,dlogm,ngdash(j))
                ngdash(j) = sqrt(ngdash(j))

                !!! TEMPORARY OUTPUT !!!
              !  if (r(j)<40.0.and.r(j)>39.0) then

!                    open(UNIT=12,FILE="halohalo_dump_steven.txt")
!                    write(*,*) "USING R = ", r(j)
!                    do i=1,nm
!                        call simps(dlogm, integrand_m(1:i),val)
!                        write(12,*) val/ngdash(j), m(i),dndm(i), ntot(i), bias(i), scale_dep_bias(dmcorr(j))*bias(i)
!                    end do
!                    close(12)
!                end if
                !!!!!!!

                if(exc_type==4)then !!ng-matched

                    !! GET INDEX WHERE ng_integral goes bigger than ng
                    val = ng_integrand(1)*dlogm/2
                    do i=2,nm-1
                        val = val + ng_integrand(i)*dlogm/2
                        if (val.ge.ngdash(j)) exit
                        val = val + ng_integrand(i)*dlogm/2
                    end do
                    if (i==nm-1)then
                        val = val + ng_integrand(nm)*dlogm/2
                        i = nm
                    end if
                    final_index = i

                    preval = val - (ng_integrand(i) + ng_integrand(i-1))*dlogm/2
                    mlim = m(final_index-1) + (m(final_index)-m(final_index-1))*(ngdash(j)-preval)/(val-preval)
!                    write(*,*) "ngdash: ", r(j), ngdash(j), log10(mlim), (ng_integrand(i) + ng_integrand(i-1))*dlogm/2
                else
                    prob = prob/integprod
                end if
            else if(exc_type==3)then
                mlim = virial_mass(r(j)/2,rhob,dhalo)
                ngdash(j) = ng_dash(m,ntot,dndm,mlim)
                !write(*,*)"ngdash", r(j),ngdash(j)/nbar
            end if

            do i=1,nk
                if(exc_type==4.or.exc_type==3)then
                    where(m.le.mlim) integrand = integrand_m * u(:,i)
                else
                    integrand = integrand_m * u(:,i)
                end if


                if(exc_type==5)then
                    do k=1,nm
                        integprod2(:,k) =  integrand(k)*integrand
                    end do
                    prob = prob * integprod2

                    call dblsimps(prob,nm,nm,dlogm,dlogm,pg2h(i,j))
                else
                    call simps(dlogm,integrand,pg2h(i,j))
                end if


            end do

            pg2h(:,j) = dmpower * pg2h(:,j)**2 / nbar**2
        end do
        !$OMP end parallel do
        if ((exc_type==1.or.exc_type==2).and..not.scale_dep)then
            call power_to_corr(nr,nk,r,pg2h,lnk,myN,myh,corr)
        else
            call power_to_corr_matrix(nr,nk,r,pg2h,lnk,myN,myh,corr)
        end if

        if(exc_type.ge.3)then
            corr = (ngdash/nbar)**2*(1.d0+corr)-1.d0
        end if

    end subroutine

    ! ============== UTILITIES =================================================
subroutine dblsimps(X,nx,ny, dx, dy,out)
        integer, intent(in) :: nx,ny
        real(8), intent(in) :: X(nx,ny)
        real(8), intent(in) :: dx, dy
        real(8), intent(out) :: out

        integer :: lengthx,lengthy
        integer :: i
        real(8),allocatable :: W(:,:)

        if (mod(nx,2) .eq. 0)then
            lengthx = nx-1
        else
            lengthx = nx
        end if

        if (mod(ny,2) .eq. 0)then
            lengthy = ny-1
        else
            lengthy = ny
        end if

        allocate(W(lengthx,lengthy))
        W = 1.d0

        W(2:lengthx-1:2, :) = W(2:lengthx-1:2, :)*4
        W(:, 2:lengthy-1:2) = W(:, 2:lengthy-1:2)*4
        W(3:lengthx-2:2, :) = W(3:lengthx-2:2, :)*2
        W(:, 3:lengthy-2:2) = W(:, 3:lengthy-2:2)*2

        out =dx * dy * sum(W * X(1:lengthx,1:lengthy)) / 9.0

    end subroutine dblsimps

    subroutine simps(dx, func,val)
        implicit none
        !!simps works on functions defined at EQUIDISTANT values of x only

        real(8), intent(in) :: dx !The grid spacing
        real(8), intent(in) :: func(:) !The function
        real(8), intent(out) :: val !The value of the integral

        integer :: n !length of func
        integer :: i,end
        real(8) :: endbit

        n = size(func)
        if(n==1)then
            val = 0.d0
            return
        else if (n==2) then
            val = (func(1) + func(2))*dx/2
        else
            if (mod(n,2)==0) then
                end =  n-1
                endbit = (func(n)+func(n-1))*dx/2
            else
                end=n
                endbit=0.d0
            end if

            val = sum(func(1:end-2:2) + func(3:end:2) + 4*func(2:end-1:2))
            val = val*dx/3 + endbit
        end if
    end subroutine simps
end module
